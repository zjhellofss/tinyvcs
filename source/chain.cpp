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

#include "player.h"
#include "iutils.h"
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

    std::thread t3([this]() {
      this->Show();
    });
    threads_.push_back(std::move(t3));
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
      LOG(FATAL) << "can not connect to " << subscription;
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

void VideoStream::Show() {
  while (true) {
    Frame f;
    for (;;) {
      bool success = show_frames_.pop(f);
      if (success) {
        break;
      }

      if (!this->player_->is_runnable()) {
        break;
      }
    }
    if (!this->player_->is_runnable() && f.detections_.empty()) {
      break;
    }
    auto detections = f.detections_;
//    cv::Mat image = f.image_;
//    for (const auto &detection : detections) {
//      cv::rectangle(image, detection.box, cv::Scalar(255, 0, 0), 8);
//    }
  }
}

void VideoStream::Infer() {
  std::vector<cv::Mat> images;
  std::vector<Frame> frames;
  while (true) {
    for (;;) {
      Frame f;
      bool read_success = frames_.pop(f);
      if (read_success) {
        images.push_back(f.preprocess_image_);
        frames.push_back(f);
        break;
      }

      if (!this->player_->is_runnable()) {
        break;
      }
    }

    if (!this->player_->is_runnable() && images.size() != batch_) {
      LOG(ERROR) << "image number do not equal batch and this player quited!";
      break;
    }

    if (inference_ && images.size() == batch_) {
      std::vector<std::vector<Detection>> detections = inference_->Infer(images, 0.2f, 0.2f);
      LOG(INFO) << "stream id: " << stream_id_ << " remain frames: " << 1024 - frames_.write_available();
      for (auto tup : boost::combine(detections, frames)) {
        std::vector<Detection> detection;
        Frame frame;
        boost::tie(detection, frame) = tup;
        frame.set_detections(detection);
        show_frames_.push(frame);
      }
      images.clear();
      frames.clear();
    }
  }
  LOG(ERROR) << "infer process is exited!";
}

void VideoStream::ReadImages() {
  uint64_t index_frame = 0;
  while (true) {
    std::optional<Frame> frame_opt = player_->get_image();
    if (frame_opt.has_value()) {
      Frame f = frame_opt.value();
      cv::Mat preprocess_image;
      auto image = f.image_;
      index_frame += 1;
      if (index_frame % duration_) {
        continue;
      }
      if (this->inference_) {
        letterbox(image, preprocess_image);
        preprocess_image.convertTo(preprocess_image, CV_32FC3, 1 / 255.0);
        f.set_preprocess_image(preprocess_image);
        frames_.push(f); //fixme push success?
      }
    } else {
      if (!player_->is_runnable())
        break;
    }
  }
  LOG(ERROR) << "read images process is exited!";
}

void VideoStream::set_inference(size_t batch, const std::string &engine_file) {
  batch_ = batch;
  this->inference_ = std::make_unique<Inference>("", engine_file, 0, true);
  this->inference_->Init();
}


