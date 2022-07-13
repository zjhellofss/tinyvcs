//
// Created by fss on 22-6-8.
//

#ifndef TINYVCS_INCLUDE_PLAYER_H_
#define TINYVCS_INCLUDE_PLAYER_H_
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/circular_buffer.hpp>

#include "sync_queue.h"
#include "ffmpeg.h"
#include "frame.h"

class Player : private boost::noncopyable {

 public:
  explicit Player(int stream_idx, std::string rtsp);

  ~Player();

  bool Open();

  void Run();

  size_t number_packet_remain();

  size_t number_decode_remain();

  int fps();

  std::optional<Frame> get_image();

 public:
  const std::string &get_rtsp() const {
    return this->input_rtsp_;
  }

  bool is_runnable() const {
    return this->is_runnable_;
  }

  void exit_loop() {
    this->is_runnable_ = false;
  }

  float mean_decode_costs();

 private:
  void ReadPackets(); ///

  void DecodePackets(); ///

  int HwDecoderInit(AVCodecContext *ctx, AVHWDeviceType type);

  static AVPixelFormat GetHwFormat(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

 private:
  boost::circular_buffer<long> decode_costs_{512};
  std::vector<std::thread> threads_;
  const std::string input_rtsp_;
  AVPixelFormat src_pixel_fmt{};
  AVFormatContext *fmt_ctx_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVDictionary *fmt_opts_ = nullptr;
  AVDictionary *codec_opts_ = nullptr;
  int video_stream_index_ = -1;

  int dw_ = 0;
  int dh_ = 0;
  AVBufferRef *hw_device_ctx_ = nullptr;
  int stream_idx_ = 0;
  std::atomic_bool is_runnable_ = false;
  SynchronizedQueue<std::shared_ptr<AVPacket>, 1024> frames_;
  SynchronizedQueue<Frame, 1024> decoded_images_;
 public:
  int64_t block_starttime_ = time(nullptr);
  int64_t block_timeout_ = 10;
  static AVPixelFormat hwformat_;
};
#endif //TINYVCS_INCLUDE_PLAYER_H_
