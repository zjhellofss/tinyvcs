//
// Created by fss on 22-6-8.
//

#ifndef TINYVCS_INCLUDE_PLAYER_H_
#define TINYVCS_INCLUDE_PLAYER_H_
#include <string>
#include <vector>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <atomic>

#include "safevec.h"
#include "ffmpeg.h"
#include "frame.h"

class Player {

 public:
  explicit Player(int stream_idx, std::string rtsp);

  ~Player();

  bool Open();

  void Run();

  std::optional<Frame> get_image();

 public:
  const std::string &get_rtsp() const {
    return this->input_rtsp_;
  }

  bool is_runnable() const {
    return this->is_runnable_;
  }
 private:
  void ReadPackets(); ///

  void DecodePackets(); ///

  int HwDecoderInit(AVCodecContext *ctx, AVHWDeviceType type);

  static AVPixelFormat GetHwFormat(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

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
  int linesize_[4]{0, 0, 0, 0};
  int video_stream_index_ = -1;
  int audio_stream_index_ = -1;
  int subtitle_stream_index_ = -1;
  int video_time_base_num_ = -1;
  int video_time_base_den_ = -1;
  int fps_ = 0;
  int64_t duration_ = 0;
  int64_t start_time_ = 0;
  int dw_ = 0;
  int dh_ = 0;
  AVBufferRef *hw_device_ctx_ = nullptr;
  int stream_idx_ = 0;
  std::atomic_bool is_runnable_ = false;
  boost::lockfree::spsc_queue<std::shared_ptr<AVPacket>, boost::lockfree::capacity<1024>> frames_;
  boost::lockfree::spsc_queue<Frame, boost::lockfree::capacity<1024>> decoded_images_;
 public:
  int64_t block_starttime_ = time(nullptr);
  int64_t block_timeout_ = 10;
  static AVPixelFormat hwformat_;
};
#endif //TINYVCS_INCLUDE_PLAYER_H_
