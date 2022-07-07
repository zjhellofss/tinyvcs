//
// Created by fss on 22-6-9.
//

#ifndef TINYVCS_INCLUDE_CHAIN_H_
#define TINYVCS_INCLUDE_CHAIN_H_
#include <string>
#include <vector>
#include <thread>

#include "websocket/client.h"

#include "tensorrt/engine.h"
#include "sync_queue.h"
#include "player.h"
#include "infer.h"

class VideoStream {
 public:
  explicit VideoStream(int stream_id,
                       int duration,
                       std::string rtsp_address,
                       std::vector<std::string> subscriptions)
      : stream_id_(stream_id),
        duration_(duration),
        rtsp_address_(std::move(rtsp_address)),
        subscriptions_(std::move(subscriptions)) {

  }
  ~VideoStream() {
    for (std::thread &t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  bool Open();

  void Show();

  void ReadImages();

  void Infer();

  void set_inference(size_t batch, const std::string &engine_file);

  void Run();

 private:
  int stream_id_ = 0;
  size_t batch_ = 0;
  int duration_ = 0;
  std::string rtsp_address_;
  std::vector<std::string> subscriptions_;
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<Connection>> connnections_;
  SynchronizedQueue<Frame, 1024> frames_;
  SynchronizedQueue<Frame, 1024> show_frames_;
  std::shared_ptr<Player> player_;
  std::unique_ptr<Inference> inference_;
};

#endif //TINYVCS_INCLUDE_CHAIN_H_
