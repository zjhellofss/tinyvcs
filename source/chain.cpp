#include <vector>
#include <string>
#include "fmt/core.h"
#include "player.h"
#include "websocket/client.h"
#include "json/jsonbuilder.h"
#include "glog/logging.h"
#include "chain.h"

void VideoStream::Run() {
  std::thread t([this]() {
    this->ReadImages();
  });
  threads_.push_back(std::move(t));
}

bool VideoStream::Open() {
  player_ = std::make_shared<Player>(stream_id_, rtsp_address_);
  bool open_success = player_->Open();
  // open websockets
  for (const std::string &subscription : subscriptions_) {
    std::shared_ptr<Connection> connection = std::make_shared<Connection>();
    bool is_open = connection->Connect(subscription);
    if (!is_open) {
      LOG(FATAL) << "Can not connect to " << subscription;
      return false;
    } else {
      connnections_.push_back(connection);
    }
  }
  if (!open_success) {
    return false;
  }
  player_->Run();
  return true;
}
void VideoStream::ReadImages() {
  while (true) {
    try {
      Frame frame = player_->GetImage();
      cv::Mat image = frame.image_;
      if (!image.empty()) {
        int height = image.size().height;
        int width = image.size().width;
        vmaps values;
        values.insert({"pts", frame.pts_});
        values.insert({"width", width});
        values.insert({"height", height});
        std::string json_res = create_json(values);
        for (const auto &connection : this->connnections_) {
          if (connection->IsRunnable()) {
            bool send_success = connection->Send(json_res);
            if (send_success) {
            }
          }
        }
      }
    }
    catch (SynchronizedVectorException &e) {
      if (!player_->IsRunnable()) {
        LOG(ERROR) << "Decode packet process is exited with error";
        break;
      }
    }
  }
  LOG(INFO) << "Read images process is exited!";
}



