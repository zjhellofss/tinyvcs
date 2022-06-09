//
// Created by fss on 22-6-9.
//

#ifndef TINYVCS_INCLUDE_CHAIN_H_
#define TINYVCS_INCLUDE_CHAIN_H_
class VideoStream {
 public:
  explicit VideoStream(int stream_id, std::string rtsp_address, std::vector<std::string> subscriptions) : stream_id_(
      stream_id), rtsp_address_(std::move(rtsp_address)), subscriptions_(std::move(subscriptions)) {

  }
  ~VideoStream() {
    for (std::thread &t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  bool Open();

  void ReadImages();

  void SendMessages();

  void Inferences();

  void Run();

 private:
  int stream_id_ = 0;
  std::string rtsp_address_;
  std::vector<std::string> subscriptions_;
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<Connection>> connnections_;
  std::shared_ptr<Player> player_;
};

#endif //TINYVCS_INCLUDE_CHAIN_H_
