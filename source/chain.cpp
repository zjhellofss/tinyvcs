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
      LOG(ERROR) << "Can not connect to " << subscription;
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
        LOG(INFO) << fmt::format("image height:{} image width:{} pts:{} ",
                                 height,
                                 width,
                                 frame.pts_);

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
      if (!player_->isRunnable()) {
        LOG(ERROR) << "Decode packet process is exited with error";
        break;
      }
    }
  }
  LOG(INFO) << "Read images process is exited!";
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
//  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  std::vector<std::string> subscriptions;
  subscriptions.emplace_back("ws://127.0.0.1:9002/");
  VideoStream stream(0, "rtsp://127.0.0.1:8554/mystream", subscriptions);
  bool b = stream.Open();
  stream.Run();
  return 0;
}

