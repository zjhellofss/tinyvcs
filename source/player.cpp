//
// Created by fss on 22-6-6.
//
#include "player.h"
#include <string>
#include <utility>
#include <memory>
#include <thread>

#include "opencv2/opencv.hpp"
#include "opencv2/cudawarping.hpp"
#include "cuda/hw_decode.h"
#include "glog/logging.h"
#include "fmt/core.h"
#include "tick.h"

#include "ffmpeg.h"
#include "frame.h"
#include "image_utils.h"
#include "convert.h"

static boost::circular_buffer<std::shared_ptr<AVPacket>> packets_pool_;
AVPixelFormat  Player::hwformat_ = AVPixelFormat::AV_PIX_FMT_NONE;

static int interruptCallback(void *opaque);

static void packetDeleter(AVPacket *packet) {

  if (packet != nullptr) {
    if (!packets_pool_.full()) {
      packets_pool_.push_back(std::shared_ptr<AVPacket>(packet, packetDeleter));
    } else {
      av_packet_unref(packet);
      av_packet_free(&packet);
      packet = nullptr;
    }
  }
}

static bool isKey(const std::shared_ptr<AVPacket> &pkt) {
  return (pkt->flags & AV_PKT_FLAG_KEY) != 0;
}

bool isCorrupted(const std::shared_ptr<AVPacket> &pkt) {
  return (bool) ((uint32_t) pkt->flags & (uint32_t) AV_PKT_FLAG_CORRUPT);
}

static void frameDeleter(AVFrame *frame) {
  if (frame != nullptr) {
    av_frame_unref(frame);
    av_frame_free(&frame);
    frame = nullptr;
  }
}

void Player::ReadPackets() {
  while (true) {
    if (!is_runnable_) {
      break;
    }
    fmt_ctx_->interrupt_callback.callback = interruptCallback;
    fmt_ctx_->interrupt_callback.opaque = this;
    block_starttime_ = time(nullptr);
    std::shared_ptr<AVPacket> packet;
    if (!packets_pool_.empty()) {
      packet = packets_pool_.front();
      packets_pool_.pop_front();
    } else {
      packet = std::shared_ptr<AVPacket>(av_packet_alloc(), packetDeleter);
    }
    auto packet_raw = packet.get();
    int ret = av_read_frame(fmt_ctx_, packet_raw);
    fmt_ctx_->interrupt_callback.callback = nullptr;
    if (ret != 0) {
      const int msg_len = 512;
      char err_msg[msg_len] = {0};
      av_make_error_string(err_msg, msg_len, ret);
      LOG(ERROR) << "read paket error: " << err_msg;
      if (ret == AVERROR_EOF || avio_feof(fmt_ctx_->pb)) {
        LOG(ERROR) << "media meet EOF";
      }
      break;
    }
    if (packet_raw->stream_index != video_stream_index_) {
      continue;
    }
    frames_.push(packet); //fixme push success?
  }
  is_runnable_ = false;
  LOG(ERROR) << "read packet process is exited!";
}

void Player::DecodePackets() {
  uint64_t index = 0;
  std::shared_ptr<AVFrame> frame = std::shared_ptr<AVFrame>(av_frame_alloc(), frameDeleter);
  std::shared_ptr<AVFrame> sw_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), frameDeleter);

  while (true) {
    av_frame_unref(frame.get());
    if (!is_runnable_) {
      break;
    }
    std::shared_ptr<AVPacket> packet;
    for (;;) {
      bool read_success = this->frames_.pop(packet);
      if (read_success && packet) {
        break;
      }
      if (!is_runnable_)
        break;
    }

    if (!packet) {
      break;
    }

    if (packet && isCorrupted(packet)) {
      LOG(ERROR) << "read bad packet";
      continue;
    }

//    TICK(DECODE)
    auto start = std::chrono::steady_clock::now();

    int ret = avcodec_send_packet(codec_ctx_, packet.get());
    if (ret != 0) {
      const int msg_len = 512;
      char err_msg[msg_len] = {0};
      av_make_error_string(err_msg, msg_len, ret);
      LOG(ERROR) << "avcodec_send_packet error: " << err_msg;
      break;
    }

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

    int width = frame->width;
    int height = frame->height;

    if (frame->pts == AV_NOPTS_VALUE) {
      continue;
    }
    if (width == 0 || height == 0 || width != dw_ || height != dh_) {
      LOG(ERROR) << "stream id: " << stream_idx_ << " frame don't have correct size index: " << frame->pts;
      continue;
    }

    std::optional<cv::cuda::GpuMat> image_gpu_opt = ConvertFrame(frame.get());
    if (!image_gpu_opt.has_value()) {
      LOG(ERROR) << "convert frame failed";
      continue;
    }
    cv::cuda::GpuMat image_gpu_yuv = image_gpu_opt.value();
    cv::cuda::GpuMat image_gpu = cv::cuda::createContinuous(image_gpu_yuv.rows, image_gpu_yuv.cols, CV_8UC3);
    convertYUV(image_gpu_yuv.data,
               height,
               width,
               width,
               image_gpu.data,
               width * 3);

    cv::cuda::resize(image_gpu, image_gpu, cv::Size(960, 640));
    Frame f(isKey(packet), frame->pts, frame->pkt_dts, index, frame->height, frame->width, image_gpu);

    auto end = std::chrono::steady_clock::now();
    LOG(INFO) << f.to_string() << " decode costs:"
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<" ms";
    this->decoded_images_.push(f);
    index += 1;
//    TOCK(DECODE)
  }

  is_runnable_ = false;
  LOG(ERROR) << "decode packet process is exited!";
}

void Player::Run() {
  is_runnable_ = true;
  std::thread t1([this]() {
    this->ReadPackets();
  });
  threads_.push_back(std::move(t1));

  std::thread t2([this]() {
    this->DecodePackets();
  });
  threads_.push_back(std::move(t2));
}

Player::Player(int stream_idx, std::string rtsp)
    : input_rtsp_(std::move(rtsp)), stream_idx_(stream_idx) {
}

AVPixelFormat Player::GetHwFormat(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  const AVPixelFormat *p;
  for (p = pix_fmts; *p != -1; p++) {
    if (*p == hwformat_)
      return *p;
  }
  return AV_PIX_FMT_NONE;
}

bool Player::Open() {
  // find cuda support
  AVHWDeviceType hw_type = av_hwdevice_find_type_by_name("cuda");
  bool has_cuda = true;
  if (hw_type == AV_HWDEVICE_TYPE_NONE) {
    std::stringstream ss;
    ss << fmt::format("device type %s is not supported.\n", "cuda");
    ss << std::string("available device types:");
    while ((hw_type = av_hwdevice_iterate_types(hw_type)) != AV_HWDEVICE_TYPE_NONE)
      ss << fmt::format(" %s", av_hwdevice_get_type_name(hw_type));
    ss << "\n";
    std::string error_message = ss.str();
    LOG(ERROR) << error_message;
    has_cuda = false;
  }

  fmt_ctx_ = avformat_alloc_context();
  if (fmt_ctx_ == nullptr) {
    LOG(FATAL) << "avformat_alloc_context failed";
    return false;
  }
  av_dict_set(&fmt_opts_, "rtsp_transport", "tcp", 0);
  av_dict_set(&fmt_opts_, "stimeout", "5000000", 0);   // us
  av_dict_set(&fmt_opts_, "buffer_size", "2048000", 0);

  fmt_ctx_->interrupt_callback.callback = interruptCallback;
  fmt_ctx_->interrupt_callback.opaque = this;
  block_starttime_ = time(nullptr);
  AVInputFormat *ifmt = nullptr;
  int ret = avformat_open_input(&fmt_ctx_, input_rtsp_.c_str(), ifmt, &fmt_opts_);
  if (ret != 0) {
    LOG(FATAL) << fmt::format("open input file[{}] failed: {}", input_rtsp_.c_str(), ret);
    return false;
  }
  fmt_ctx_->interrupt_callback.callback = nullptr;

  ret = avformat_find_stream_info(fmt_ctx_, nullptr);
  if (ret != 0) {
    LOG(FATAL) << fmt::format("can not find stream: {}", ret);
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
    LOG(FATAL) << ("Can not find video stream");
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
    LOG(FATAL) << "can not find decoder " << avcodec_get_name(codec_param->codec_id);
    return false;
  }

  LOG(INFO) << fmt::format("codec_name: {}=>{}", codec->name, codec->long_name);

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (codec_ctx_ == nullptr) {
    LOG(FATAL) << "avcodec_alloc_context3";
    return false;
  }

  if (has_cuda) {
    for (int i = 0;; i++) {
      const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
      if (!config) {
        char error_buf[512] = {0};
        snprintf(error_buf, 512, "decoder %s does not support device type %s",
                 codec->name, av_hwdevice_get_type_name(hw_type));
        LOG(FATAL) << error_buf;
        return false;
      }
      if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
          config->device_type == hw_type) {
        Player::hwformat_ = config->pix_fmt;
        break;
      }
    }
  }

  ret = avcodec_parameters_to_context(codec_ctx_, codec_param);
  if (ret != 0) {
    LOG(FATAL) << fmt::format("avcodec_parameters_to_context error: {}", ret);
    return false;
  }

  if (has_cuda) {
    codec_ctx_->get_format = GetHwFormat;
    if (HwDecoderInit(codec_ctx_, hw_type) < 0)
      LOG(FATAL) << "HW decoder init error";
  }

  if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO || codec_ctx_->codec_type == AVMEDIA_TYPE_AUDIO) {
    av_dict_set(&codec_opts_, "refcounted_frames", "1", 0);
  }
  ret = avcodec_open2(codec_ctx_, codec, &codec_opts_);
  if (ret != 0) {
    LOG(FATAL) << fmt::format("can not open software codec error: {}", ret);
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
    LOG(FATAL) << "get pixel format error";
    return false;
  }
  dw_ = sw >> 2 << 2;
  dh_ = sh;
  dst_pixel_fmt = AV_PIX_FMT_BGR24;
  sws_context_ = sws_getContext(sw, sh, src_pixel_fmt, dw_, dh_, dst_pixel_fmt, SWS_BICUBIC,
                                nullptr, nullptr, nullptr);
  if (!sws_context_) {
    LOG(FATAL) << "sws get context failed";
    return false;
  }
  linesize_[0] = dw_ * 3;
  return true;
}

int Player::HwDecoderInit(AVCodecContext *ctx, AVHWDeviceType type) {
  int err;
  err = av_hwdevice_ctx_create(&hw_device_ctx_, type,
                               nullptr, nullptr, 0);
  if (err != 0) {
    return err;
  }
  ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  return err;
}

Player::~Player() {
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

  if (hw_device_ctx_ != nullptr) {
    av_buffer_unref(&hw_device_ctx_);
    hw_device_ctx_ = nullptr;
  }
}

std::optional<Frame> Player::get_image() {
  Frame frame;
  for (;;) {
    bool has_frame = this->decoded_images_.pop(frame);
    if (has_frame) {
      return frame;
    }

    if (!is_runnable_) {
      break;
    }
  }
  return std::nullopt;
}

static int interruptCallback(void *opaque) {
  if (opaque == nullptr) return 0;
  Player *player = (Player *) opaque;
  if (time(nullptr) - player->block_starttime_ > player->block_timeout_) {
    std::string s = fmt::format("interrupt quit, media address={}", player->get_rtsp());
    LOG(ERROR) << s;
    return 1;
  }
  return 0;
}
