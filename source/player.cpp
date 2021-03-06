//
// Created by fss on 22-6-6.
//
#include "player.h"
#include <string>
#include <utility>
#include <memory>
#include <thread>

#include "opencv2/cudawarping.hpp"
#include "cuda/convert_frame.h"
#include "glog/logging.h"
#include "fmt/core.h"

#include "ffmpeg.h"
#include "frame.h"
#include "cuda/convert_yuv.h"
#include "cuda/op/preprocess_op.h"
#include "cuda/op/letterbox_op.h"
#include "cuda/op/avframe_op.h"

AVPixelFormat  Player::hwformat_ = AVPixelFormat::AV_PIX_FMT_NONE;

static int interruptCallback(void *opaque);

static void packetDeleter(AVPacket *packet) {
  if (packet != nullptr) {
    av_packet_unref(packet);
    av_packet_free(&packet);
    packet = nullptr;
  }
}

static bool isKey(const std::shared_ptr<AVPacket> &pkt) {
  return (pkt->flags & AV_PKT_FLAG_KEY) != 0;
}

static bool isCorrupted(const std::shared_ptr<AVPacket> &pkt) {
  return (bool) ((uint32_t) pkt->flags & (uint32_t) AV_PKT_FLAG_CORRUPT);
}

static std::string createErrorBuf(int err_num) {
  const int msg_len = 512;
  std::string err_msg;
  err_msg.reserve(msg_len);
  av_make_error_string(err_msg.data(), msg_len, err_num);
  return err_msg;
}

static void frameDeleter(AVFrame *frame) {
  if (frame != nullptr) {
    av_frame_unref(frame);
    av_frame_free(&frame);
    frame = nullptr;
  }
}

static int64_t ptsToTimeStamp(int64_t packet_pts, const AVStream *video_stream) {
  if (!video_stream) {
    return 0;
  }

  if (video_stream->start_time != AV_NOPTS_VALUE) {
    return (packet_pts - video_stream->start_time) * 1000 * video_stream->time_base.num / video_stream->time_base.den;
  }
  return packet_pts * 1000 * video_stream->time_base.num / video_stream->time_base.den;
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
    packet = std::shared_ptr<AVPacket>(av_packet_alloc(), packetDeleter);

    auto packet_raw = packet.get();
    int ret = av_read_frame(fmt_ctx_, packet_raw);
    fmt_ctx_->interrupt_callback.callback = nullptr;
    if (ret != 0) {
      std::string err_msg = createErrorBuf(ret);
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
      std::string err_msg = createErrorBuf(ret);
      LOG(ERROR) << "avcodec_send_packet error: " << err_msg;
      break;
    }

    ret = avcodec_receive_frame(codec_ctx_, frame.get());
    if (ret != 0) {
      std::string err_msg = createErrorBuf(ret);
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

    FrameConvertOp frame_op(frame);
    cv::cuda::GpuMat yuv_image;
    frame_op.Process(yuv_image);
    if (yuv_image.empty()) {
      LOG(ERROR) << "convert frame failed";
      continue;
    }
    PreProcessOp pre_op(height, width, 640, 960);
    cv::cuda::GpuMat image_gpu;
    pre_op.Process(yuv_image, image_gpu);

    LetterBoxOp letter_box_op(640, 640);
    cv::cuda::GpuMat preprocess_image;
    letter_box_op.Process(image_gpu, preprocess_image);

    Frame f(image_gpu, preprocess_image,
            width,
            height,
            isKey(packet),
            packet->pts,
            packet->dts,
            ptsToTimeStamp(packet->pts, this->fmt_ctx_->streams[video_stream_index_]), index);

    auto end = std::chrono::steady_clock::now();
    auto decode_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    this->decode_costs_.push_back(decode_cost);
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
    LOG(ERROR) << "avformat_alloc_context failed";
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
    LOG(ERROR) << fmt::format("open input file[{}] failed: {}", input_rtsp_.c_str(), ret);
    return false;
  }
  fmt_ctx_->interrupt_callback.callback = nullptr;

  ret = avformat_find_stream_info(fmt_ctx_, nullptr);
  if (ret != 0) {
    LOG(ERROR) << fmt::format("can not find stream: {}", ret);
    return false;
  }
  LOG(INFO) << fmt::format("stream_num={}", fmt_ctx_->nb_streams);

  video_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  LOG(INFO) << fmt::format("video_stream_index={:d}", video_stream_index_);
  if (video_stream_index_ < 0) {
    LOG(ERROR) << ("Can not find video stream");
    return false;
  }

  AVStream *video_stream = fmt_ctx_->streams[video_stream_index_];
  LOG(INFO) << fmt::format("video_stream time_base={}/{}", video_stream->time_base.num, video_stream->time_base.den);

  AVCodecParameters *codec_param = video_stream->codecpar;
  LOG(INFO) << fmt::format("codec_id={}:{}", codec_param->codec_id, avcodec_get_name(codec_param->codec_id));

  const AVCodec *codec = avcodec_find_decoder(codec_param->codec_id);
  if (codec == nullptr) {
    LOG(ERROR) << "can not find decoder " << avcodec_get_name(codec_param->codec_id);
    return false;
  }

  LOG(INFO) << fmt::format("codec_name: {}=>{}", codec->name, codec->long_name);
  codec_ctx_ = avcodec_alloc_context3(codec);
  if (codec_ctx_ == nullptr) {
    LOG(ERROR) << "avcodec_alloc_context3";
    return false;
  }

  if (has_cuda) {
    for (int i = 0;; i++) {
      const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
      if (!config) {
        char error_buf[512] = {0};
        snprintf(error_buf, 512, "decoder %s does not support device type %s",
                 codec->name, av_hwdevice_get_type_name(hw_type));
        LOG(ERROR) << error_buf;
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
    LOG(ERROR) << fmt::format("avcodec_parameters_to_context error: {}", ret);
    return false;
  }

  if (has_cuda) {
    codec_ctx_->get_format = GetHwFormat;
    if (HwDecoderInit(codec_ctx_, hw_type) < 0) {
      LOG(ERROR) << "HW decoder init error";
      return false;
    }
  }

  if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO || codec_ctx_->codec_type == AVMEDIA_TYPE_AUDIO) {
    av_dict_set(&codec_opts_, "refcounted_frames", "1", 0);
  }
  ret = avcodec_open2(codec_ctx_, codec, &codec_opts_);
  if (ret != 0) {
    LOG(ERROR) << fmt::format("can not open software codec error: {}", ret);
    return false;
  }

  video_stream->discard = AVDISCARD_DEFAULT;

  dw_ = codec_ctx_->width;
  dh_ = codec_ctx_->height;
  src_pixel_fmt = codec_ctx_->pix_fmt;
  if (dw_ <= 0 || dh_ <= 0 || src_pixel_fmt == AV_PIX_FMT_NONE) {
    LOG(ERROR) << "get pixel format error";
    return false;
  }

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

size_t Player::number_decode_remain() {
  return this->decoded_images_.read_available();
}

size_t Player::number_packet_remain() {
  return this->frames_.read_available();
}

int Player::fps() {
  AVStream *video_stream = this->fmt_ctx_->streams[video_stream_index_];
  if (video_stream->avg_frame_rate.num && video_stream->avg_frame_rate.den) {
    int fps = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    return fps;
  }
  return -1;
}

float Player::mean_decode_costs() {
  if (this->decode_costs_.empty()) {
    return 0.f;
  }
  long sum = std::accumulate(this->decode_costs_.begin(), this->decode_costs_.end(), 0l);
  float mean = (float) sum / (float) this->decode_costs_.size();
  return mean;
}

static int interruptCallback(void *opaque) {
  if (opaque == nullptr) return 0;
  Player *player = (Player *) opaque;
  if (player->is_runnable() && time(nullptr) - player->block_starttime_ > player->block_timeout_) {
    std::string s = fmt::format("timeout interrupt quit, media address={}", player->get_rtsp());
    LOG(ERROR) << s;
    return 1;
  }
  return 0;
}
