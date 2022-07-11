//
// Created by fss on 22-6-6.
//
#include "frame.h"
#include "glog/logging.h"

void Frame::set_preprocess_image(const cv::cuda::GpuMat &image) {
  this->preprocess_image_ = image;
}

void Frame::set_detections(const std::vector<Detection> &detections) {
  this->detections_ = detections;
}

std::string Frame::to_string() {
  std::string
      message = fmt::format("idx:{} pts:{} dts:{} is_key:{} width:{} height:{} time:{}",
                            this->index_,
                            this->pts_,
                            this->dts_,
                            this->is_key_,
                            this->width_,
                            this->height_,
                            this->timestamp_);
  return message;
}
Frame::Frame(const cv::cuda::GpuMat &image,
             int width,
             int height,
             bool is_key,
             uint64_t pts,
             uint64_t dts,
             uint64_t timestamp,
             uint64_t index)
    : gpu_image_(image),
      width_(width),
      height_(height),
      is_key_(is_key),
      pts_(pts),
      dts_(dts),
      timestamp_(timestamp),
      index_(index) {

}

void Frame::set_cpu_image() {
  if (!this->cpu_image_.empty() || this->gpu_image_.empty())
    return;
  this->gpu_image_.download(this->cpu_image_);
}

