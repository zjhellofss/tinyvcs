//
// Created by fss on 22-6-6.
//
#include "ffmpeg_utils.h"
#include "glog/logging.h"
#include "fmt/core.h"
#include "opencv2/opencv.hpp"
#include "frame.h"
#include "safe_vector.h"

#include <string>
#include <utility>
#include <memory>
#include <optional>
#include <thread>
#include <cassert>

static int InterruptCallback(void *opaque);

static void PacketDeleter(AVPacket *packet) {
  if (packet != nullptr) {
    av_packet_unref(packet);
    av_packet_free(&packet);
    packet = nullptr;
  }
}

static void FrameDeleter(AVFrame *frame) {
  if (frame != nullptr) {
    av_frame_unref(frame);
    av_frame_free(&frame);
    frame = nullptr;
  }
}

class Player {

 public:
  explicit Player(int stream_idx, std::string rtsp);

  ~Player() {
    for (std::thread &t : this->threads_) {
      if (t.joinable()) {
        t.join();
      }
    }

    if (fmt_opts_) {
      av_dict_free(&fmt_opts_);
      fmt_opts_ = nullptr;
    }

    if (codec_opts_) {
      av_dict_free(&codec_opts_);
      codec_opts_ = nullptr;
    }

    if (codec_ctx_) {
      avcodec_close(codec_ctx_);
      avcodec_free_context(&codec_ctx_);
      codec_ctx_ = nullptr;
    }

    if (fmt_ctx_) {
      avformat_close_input(&fmt_ctx_);
      avformat_free_context(fmt_ctx_);
      fmt_ctx_ = nullptr;
    }

    if (sws_context_) {
      sws_freeContext(sws_context_);
      sws_context_ = nullptr;
    }

    if (this->data_[0] != nullptr) {
      unsigned char *p = this->data_[0];
      delete[]p;
    }

  }
  bool Open() {
    fmt_ctx_ = avformat_alloc_context();
    if (fmt_ctx_ == nullptr) {
      LOG(INFO) << "avformat_alloc_context failed";
      return false;
    }
    av_dict_set(&fmt_opts_, "rtsp_transport", "tcp", 0);
    av_dict_set(&fmt_opts_, "stimeout", "5000000", 0);   // us
    av_dict_set(&fmt_opts_, "buffer_size", "2048000", 0);

    fmt_ctx_->interrupt_callback.callback = InterruptCallback;
    fmt_ctx_->interrupt_callback.opaque = this;
    block_starttime_ = time(nullptr);
    AVInputFormat *ifmt = nullptr;
    int ret = avformat_open_input(&fmt_ctx_, input_rtsp_.c_str(), ifmt, &fmt_opts_);
    if (ret != 0) {
      LOG(ERROR) << fmt::format("Open input file[{}] failed: {}", input_rtsp_.c_str(), ret);
      return false;
    }
    fmt_ctx_->interrupt_callback.callback = nullptr;

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret != 0) {
      LOG(ERROR) << fmt::format("Can not find stream: {}", ret);
      return false;
    }
    LOG(INFO) << fmt::format("stream_num={}", fmt_ctx_->nb_streams);

    video_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    audio_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    subtitle_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_SUBTITLE, -1, -1, nullptr, 0);
    LOG(INFO) << fmt::format("video_stream_index={:d}", video_stream_index_);
    LOG(INFO) << fmt::format("audio_stream_index={:d}", audio_stream_index_);
    LOG(INFO) << fmt::format("subtitle_stream_index={:d}", subtitle_stream_index_);
    if (video_stream_index_ < 0) {
      LOG(ERROR) << ("Can not find video stream");
      return false;
    }

    AVStream *video_stream = fmt_ctx_->streams[video_stream_index_];
    video_time_base_num_ = video_stream->time_base.num;
    video_time_base_den_ = video_stream->time_base.den;
    LOG(INFO) << fmt::format("video_stream time_base={}/{}", video_stream->time_base.num, video_stream->time_base.den);

    AVCodecParameters *codec_param = video_stream->codecpar;
    LOG(INFO) << fmt::format("codec_id={}:{}", codec_param->codec_id, avcodec_get_name(codec_param->codec_id));

    const AVCodec *codec = nullptr;

    codec = avcodec_find_decoder(codec_param->codec_id);
    if (codec == nullptr) {
      LOG(ERROR) << "Can not find decoder " << avcodec_get_name(codec_param->codec_id);
      return false;
    }

    LOG(INFO) << fmt::format("codec_name: {}=>{}", codec->name, codec->long_name);

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (codec_ctx_ == nullptr) {
      LOG(ERROR) << "avcodec_alloc_context3";
      return false;
    }

    ret = avcodec_parameters_to_context(codec_ctx_, codec_param);
    if (ret != 0) {
      LOG(ERROR) << fmt::format("avcodec_parameters_to_context error: {}", ret);
      return false;
    }

    if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO || codec_ctx_->codec_type == AVMEDIA_TYPE_AUDIO) {
      av_dict_set(&codec_opts_, "refcounted_frames", "1", 0);
    }
    ret = avcodec_open2(codec_ctx_, codec, &codec_opts_);
    if (ret != 0) {
      LOG(ERROR) << fmt::format("Can not open software codec error: {}", ret);
      return false;
    }
    video_stream->discard = AVDISCARD_DEFAULT;

    if (video_stream->avg_frame_rate.num && video_stream->avg_frame_rate.den) {
      fps_ = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    }
    duration_ = 0;
    start_time_ = 0;
    if (video_time_base_num_ && video_time_base_den_) {
      if (video_stream->duration > 0) {
        duration_ = video_stream->duration / (double) video_time_base_den_ * video_time_base_num_ * 1000;
      }
      if (video_stream->start_time > 0) {
        start_time_ = video_stream->start_time / (double) video_time_base_den_ * video_time_base_num_ * 1000;
      }
    }
    LOG(INFO) << fmt::format("fps={} duration={} start_time={}", fps_, duration_, start_time_);

    int sw = codec_ctx_->width;
    int sh = codec_ctx_->height;
    src_pixel_fmt = codec_ctx_->pix_fmt;
    if (sw <= 0 || sh <= 0 || src_pixel_fmt == AV_PIX_FMT_NONE) {
      LOG(ERROR) << "Get pixel format error";
      return false;
    }
    int dw = sw >> 2 << 2;
    int dh = sh;
    dst_pixel_fmt = AV_PIX_FMT_BGR24;
    sws_context_ = sws_getContext(sw, sh, src_pixel_fmt, dw, dh, dst_pixel_fmt, SWS_BICUBIC,
                                  nullptr, nullptr, nullptr);
    if (!sws_context_) {
      LOG(ERROR) << "sws_getContext failed";
      return false;
    }
    data_[0] = new unsigned char[dw * dh * 3];
    linesize_[0] = dw * 3;
    return true;
  }

  void ReadPackets(); ///

  void DecodePackets(); ///

  void Run();

 public:
  const std::string &GetRtsp() const {
    return this->input_rtsp_;
  }

 private:
  std::vector<std::thread> threads_;
  const std::string input_rtsp_;
  AVPixelFormat src_pixel_fmt{};
  AVPixelFormat dst_pixel_fmt{};
  SwsContext *sws_context_ = nullptr;
  AVFormatContext *fmt_ctx_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVDictionary *fmt_opts_ = nullptr;
  AVDictionary *codec_opts_ = nullptr;
  uint8_t *data_[4] = {nullptr, nullptr, nullptr, nullptr};
  int linesize_[4];
  int video_stream_index_ = -1;
  int audio_stream_index_ = -1;
  int subtitle_stream_index_ = -1;
  int video_time_base_num_ = -1;
  int video_time_base_den_ = -1;
  int fps_ = 0;
  int64_t duration_ = 0;
  int64_t start_time_ = 0;
  int stream_idx_ = 0;
  std::atomic_bool is_runable_ = false;
  SynchronizedVector<std::shared_ptr<AVFrame>> frames_;
 public:
  int64_t block_starttime_ = time(nullptr);
  int64_t block_timeout_ = 10;
};

void Player::ReadPackets() {
  int64_t pts = 0;
  while (true) {
    if (!is_runable_) {
      break;
    }
    fmt_ctx_->interrupt_callback.callback = InterruptCallback;
    fmt_ctx_->interrupt_callback.opaque = this;
    block_starttime_ = time(nullptr);
    auto packet = std::shared_ptr<AVPacket>(av_packet_alloc(), PacketDeleter);
    auto packet_raw = packet.get();
    av_init_packet(packet_raw);
    int ret = av_read_frame(fmt_ctx_, packet_raw);
    fmt_ctx_->interrupt_callback.callback = nullptr;
    if (ret != 0) {
      const int msg_len = 512;
      char err_msg[msg_len] = {0};
      av_make_error_string(err_msg, msg_len, ret);
      LOG(ERROR) << "Read paket error: " << err_msg;
      if (ret == AVERROR_EOF || avio_feof(fmt_ctx_->pb)) {
        LOG(INFO) << "Media meet EOF";
      }
      break;
    }
    if (packet_raw->stream_index != video_stream_index_) {
      continue;
    }
    LOG(INFO) << "packet pts: " << packet->pts;

    ret = avcodec_send_packet(codec_ctx_, packet.get());
    if (ret != 0) {
      const int msg_len = 512;
      char err_msg[msg_len] = {0};
      av_make_error_string(err_msg, msg_len, ret);
      LOG(ERROR) << "avcodec_send_packet error: " << err_msg;
      break;
    }

    std::shared_ptr<AVFrame> frame = std::shared_ptr<AVFrame>(av_frame_alloc(), FrameDeleter);
    ret = avcodec_receive_frame(codec_ctx_, frame.get());
    if (ret != 0) {
      const int msg_len = 512;
      char err_msg[msg_len] = {0};
      av_make_error_string(err_msg, msg_len, ret);
      LOG(ERROR) << "avcodec_receive_frame error: " << err_msg;
      if (ret != -EAGAIN) {
        break;
      }
    }
    LOG(INFO) << fmt::format("frame pts:{} width:{} height:{} ", frame->pts, frame->width, frame->height);
    pts += 1;
    frames_.Push(frame);
  }
  is_runable_ = false;
  LOG(INFO) << "Read packet process is exited!";
}

void Player::DecodePackets() {
  while (true) {
    if (!is_runable_) {
      break;
    }
    std::shared_ptr<AVFrame> f_ops;
    try {
      f_ops = this->frames_.Pop();
    }
    catch (std::exception &e) {
      LOG(ERROR) << e.what();
      if (!is_runable_ && this->frames_.Empty()) {
        break;
      }
    }
    if (!f_ops) {
      break;
    } else {
      AVFrame *show_frame = f_ops.get();
      int width = show_frame->width;
      int height = show_frame->height;
      if (width == 0 || height == 0) {
        continue;
      }
      cv::Mat image = cv::Mat(height, width, CV_8UC3);
      const int h = sws_scale(sws_context_, show_frame->data, show_frame->linesize, 0, height,
                              &image.data, linesize_);
      if (h < 0 || h != show_frame->height) {
        LOG(ERROR) << "sws scale convert failed " << show_frame->pts;
      } else {
        int64_t pts = show_frame->pts;
      }
    }
  }
  is_runable_ = false;
  LOG(INFO) << "Decode packet process is exited!";
}
void Player::Run() {
  is_runable_ = true;
  std::thread t1([this]() {
    this->ReadPackets();
  });
  threads_.push_back(std::move(t1));

  std::thread t2([this]() {
    this->DecodePackets();
  });
  threads_.push_back(std::move(t2));
}

Player::Player(int stream_idx, std::string rtsp) : stream_idx_(stream_idx), input_rtsp_(std::move(rtsp)) {

}

static int InterruptCallback(void *opaque) {
  if (opaque == nullptr) return 0;
  Player *player = (Player *) opaque;
  if (time(nullptr) - player->block_starttime_ > player->block_timeout_) {
    std::string s = fmt::format("interrupt quit, media address={}", player->GetRtsp());
    LOG(ERROR) << s;
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;
  Player p(0, "rtsp://127.0.0.1:8554/mystream");
  bool b = p.Open();
  assert(b);
  LOG(INFO) << "process start";
  p.Run();

  google::ShutdownGoogleLogging();
  return 0;
}