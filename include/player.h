//
// Created by fss on 22-6-8.
//

#ifndef TINYVCS_INCLUDE_PLAYER_H_
#define TINYVCS_INCLUDE_PLAYER_H_
#include "safevec.h"
#include "ffmpeg.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>

class Player {

 public:
  explicit Player(int stream_idx, std::string rtsp);

  ~Player();

  bool Open();

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
  int stream_idx_ = 0;
  std::atomic_bool is_runable_ = false;
  SynchronizedVector<std::shared_ptr<AVFrame>> frames_;
 public:
  int64_t block_starttime_ = time(nullptr);
  int64_t block_timeout_ = 10;
};
#endif //TINYVCS_INCLUDE_PLAYER_H_
