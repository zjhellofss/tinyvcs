#include "video_stream.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "fmt/core.h"
#include "boost/range/combine.hpp"
#include "glog/logging.h"

#include "image_utils.h"
#include "player.h"
#include "infer.h"

void sync_monitor_handle(const boost::system::error_code &error_code,
                         boost::asio::deadline_timer *timer,
                         VideoStream *stream) {
  std::shared_ptr<Player> player = stream->player_;
  size_t packet_remain = player->number_packet_remain();
  size_t decode_image_remain = player->number_decode_remain();
  size_t infer_image_remain = stream->frames_.read_available();
  size_t show_image_remain = stream->show_frames_.read_available();
  float mean_time_decode = player->mean_decode_costs();
  float mean_time_infer = stream->inference_->infer_costs();
  int fps = player->fps();
  std::string info =
      fmt::format(
          "fps:{} mean_decode_time:{} ms mean_infer_time:{}",
          fps,
          mean_time_decode, mean_time_infer);
  std::string remain_info = fmt::format("packet remain:{} decoded remain:{} infer remain:{} process remain:{}",
                                        packet_remain,
                                        decode_image_remain,
                                        infer_image_remain,
                                        show_image_remain);
  LOG(INFO) << info;
  LOG(INFO) << remain_info;
  if (player->is_runnable()) {
    timer->expires_at(timer->expires_at() + boost::posix_time::seconds(1));
    timer->async_wait(boost::bind(sync_monitor_handle, boost::asio::placeholders::error, timer, stream));
  }
}

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
      this->ProcessResults();
    });
    threads_.push_back(std::move(t3));
  }

  std::thread monitor([&] {
    boost::asio::io_service io_service;
    boost::asio::deadline_timer timer(io_service, boost::posix_time::seconds(1));
    timer.async_wait(boost::bind(sync_monitor_handle, boost::asio::placeholders::error(), &timer, this));
    io_service.run();
  });
  threads_.push_back(std::move(monitor));
}

bool VideoStream::Open() {
  player_ = std::make_shared<Player>(stream_id_, rtsp_address_);
  bool open_success = player_->Open();
  if (!open_success) {
    LOG(ERROR) << "player open failed";
    return false;
  }
  // open zeromq
  if (!channel_) {
    this->channel_ = std::make_shared<ClientChannel>(this->subscription_);
    open_success = this->channel_->Init();
    if (!open_success) {
      LOG(ERROR) << "zeromq open failed";
      return false;
    }
  }
  player_->Run();
  return true;
}

void VideoStream::ProcessResults() {
//  cv::namedWindow("test");
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
    f.set_cpu_image();
    f.set_stream_id(this->stream_id_);
    cv::cuda::GpuMat image_gpu = f.gpu_image_;
    f.set_cpu_image();
//    for (const auto &detection : detections) {
//      cv::rectangle(f.cpu_image_, detection.box, cv::Scalar(255, 255, 0),4);
//    }
//    cv::imshow("test", f.cpu_image_);
//    cv::waitKey(20);
  }
}

void VideoStream::Infer() {
  std::vector<cv::cuda::GpuMat> images;
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
      std::vector<std::vector<Detection>>
          detections = inference_->Infer(images, this->infer_width_, this->infer_height_, 0.45f, 0.45f);
      if (detections.size() == images.size()) {
        for (auto tup : boost::combine(detections, frames)) {
          std::vector<Detection> detection;
          Frame frame;
          boost::tie(detection, frame) = tup;
          frame.set_detections(detection);
          show_frames_.push(frame);
        }
      }
      images.clear();
      frames.clear();
    }
  }
  LOG(ERROR) << "infer process is exited!";
}

void VideoStream::ReadImages() {
  while (true) {
    std::optional<Frame> frame_opt = player_->get_image();
    if (frame_opt.has_value()) {
      Frame f = frame_opt.value();
      cv::cuda::GpuMat preprocess_image;
      auto image = f.gpu_image_;

      if (f.index_ % duration_) {
        continue;
      }
      if (this->inference_)
        frames_.push(f); //fixme push success?
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

void VideoStream::exit_loop() {
  if (this->player_) {
    this->player_->exit_loop();
  }
}

void VideoStream::PlayerMonitor() {
  while (true) {

  }
}


