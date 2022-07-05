//
// Created by fss on 22-6-6.
//
#include "frame.h"
#include <utility>
Frame::Frame(uint64_t pts, uint64_t origin_pts, cv::cuda::GpuMat image)
    : pts_(pts), origin_pts_(origin_pts), image_(std::move(image)) {}

void Frame::set_preprocess_image(const cv::cuda::GpuMat &image) {
  this->preprocess_image_ = image;
}

void Frame::set_detections(const std::vector<Detection> &detections) {
  this->detections_ = detections;
}
