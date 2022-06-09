#include <utility>
#include <vector>
#include <string>
#include "fmt/core.h"
#include "player.h"
#include "glog/logging.h"

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

  bool Open() {
    player_ = std::make_shared<Player>(stream_id_, rtsp_address_);
    bool open_success = player_->Open();
    if (!open_success) {
      return false;
    }
    player_->Run();
    return true;
  }

  void ReadImages() {
    while (true) {
      try {
        Frame frame = player_->GetImage();
        cv::Mat image = frame.image_;
        if (!image.empty()) {
          LOG(INFO) << fmt::format("image height:{} image width:{} pts:{} ",
                                   image.size().height,
                                   image.size().width,
                                   frame.pts_);
        }
      }
      catch (SynchronizedVectorException &e) {
        LOG(ERROR) << e.what();
        if (player_->isRunnable()) {
          LOG(WARNING) << "Decode process is exited";
          break;
        }
      }
    }
    LOG(INFO) << "Read images process is exited!";
  }
  void SendMessages() {

  }

  void Inferences();

  void Run() {
    std::thread t([this]() {
      this->ReadImages();
    });
    threads_.push_back(std::move(t));
  }

 private:
  int stream_id_ = 0;
  std::string rtsp_address_;
  std::vector<std::string> subscriptions_;
  std::vector<std::thread> threads_;
  std::shared_ptr<Player> player_;
};
