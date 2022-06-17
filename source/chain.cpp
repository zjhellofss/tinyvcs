#include "chain.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "fmt/core.h"
#include "websocket/client.h"
#include "boost/range/combine.hpp"
#include "json/jsonbuilder.h"
#include "glog/logging.h"
#include "tick.h"

#include "player.h"
#include "image_utils.h"
#include "infer.h"

void VideoStream::Run() {
  std::thread t1([this]() {
    this->ReadImages();
  });
  threads_.push_back(std::move(t1));

  if (this->inference_) {
    std::thread t2([this]() {
      this->Infer();
    });
    threads_.push_back(std::move(t2));
  }
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

void VideoStream::Infer() {
  std::vector<cv::Mat> images;
  while (true) {
    for (;;) {
      if (frames_.read_available()) {
        cv::Mat image;
        bool read_success = frames_.pop(image);
        if (read_success) {
          images.push_back(image);
          break;
        }
      }

      if (!this->player_->is_runnable()) {
        break;
      }
    }

    if (!this->player_->is_runnable() && images.size() != batch_) {
      break;
    }

    if (inference_ && images.size() == batch_) {
      std::vector<std::vector<Detection>> detections = inference_->Infer(images, 0.2f, 0.2f);
      LOG(INFO) << "stream id: " << stream_id_ << " remain frames: " << 1024 - frames_.write_available();
      for (auto tup : boost::combine(detections, images)) {
        std::vector<Detection> detection;
        cv::Mat image;
        boost::tie(detection, image) = tup;
        for (const auto &info : detection) {
          cv::rectangle(image, info.box, cv::Scalar(255, 0, 0), 4);
        }
        cv::imshow("window", image);
        cv::waitKey(3);
      }
      images.clear();
      if (!this->player_->is_runnable()) {
        break;
      }
    }
  }
}

void VideoStream::ReadImages() {
  uint64_t index_frame = 0;
  while (true) {
    std::optional<Frame> frame_opt = player_->get_image();
    if (frame_opt.has_value()) {
      cv::Mat preprocess_image;
      auto image = frame_opt.value().image_;
      letterbox(image, preprocess_image);
      preprocess_image.convertTo(preprocess_image, CV_32FC3, 1 / 255.0);

      index_frame += 1;
      if (index_frame % duration_ == 0) {
        continue;
      }
      if (this->inference_) {
        frames_.push(preprocess_image); //fixme push success?
      }
    } else {
      if (!player_->is_runnable())
        break;
    }
  }
  LOG(INFO) << "read images process is exited!";
  cv::destroyAllWindows();
}

void VideoStream::set_inference(size_t batch, const std::string &engine_file) {
  batch_ = batch;
  this->inference_ = std::make_unique<Inference>("", engine_file, 0, true);
  this->inference_->Init();
}


