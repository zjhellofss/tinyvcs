//
// Created by fss on 22-6-9.
//

#ifndef TINYVCS_INCLUDE_VIDEO_STREAM_H_
#define TINYVCS_INCLUDE_VIDEO_STREAM_H_
#include <string>
#include <vector>
#include <thread>
#include "boost/core/noncopyable.hpp"
#include "boost/bind.hpp"
#include "boost/asio.hpp"
#include "zeromq/client.h"
#include "tensorrt/engine.h"
#include "sync_queue.h"
#include "player.h"
#include "infer.h"

class VideoStream : private boost::noncopyable {
 public:
  explicit VideoStream(int stream_id,
                       int duration,
                       int height,
                       int width,
                       std::string rtsp_address,
                       std::string subscription)
      : stream_id_(stream_id),
        duration_(duration),
        infer_width_(width),
        infer_height_(height),
        rtsp_address_(std::move(rtsp_address)),
        subscription_(std::move(subscription)) {

  }
  ~VideoStream() {
    for (std::thread &t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  bool Open();

  void ProcessResults();

  void ReadImages();

  void Infer();

  void set_inference(size_t batch, const std::string &engine_file);

  void exit_loop();

  void Run();

  void PlayerMonitor();

  friend void sync_monitor_handle(const boost::system::error_code &error_code,
                               boost::asio::deadline_timer *timer,
                               VideoStream *stream);

 private:
  int stream_id_ = 0;
  size_t batch_ = 0;
  int duration_ = 0;
  int infer_width_ = 0;
  int infer_height_ = 0;
  std::string rtsp_address_;
  std::string subscription_;
  std::vector<std::thread> threads_;
  SynchronizedQueue<Frame, 1024> frames_;
  SynchronizedQueue<Frame, 1024> show_frames_;
  std::shared_ptr<ClientChannel> channel_;
  std::shared_ptr<Player> player_;
  std::unique_ptr<Inference> inference_;
};

#endif //TINYVCS_INCLUDE_VIDEO_STREAM_H_
