//
// Created by fss on 22-6-6.
//
#include "frame.h"
#include <utility>

Frame::Frame(bool is_key, uint64_t pts, uint64_t dts, uint64_t index, const cv::cuda::GpuMat &image)
    : is_key_(is_key), pts_(pts), dts_(dts), index_(index), image_(image) {}

void Frame::set_preprocess_image(const cv::cuda::GpuMat &image) {
  this->preprocess_image_ = image;
}

void Frame::set_detections(const std::vector<Detection> &detections) {
  this->detections_ = detections;
}

std::string Frame::to_string() {
  std::string
      message = fmt::format("idx:{} pts:{} dts:{} is_key:{}", this->index_, this->pts_, this->dts_, this->is_key_);
  return message;
}

