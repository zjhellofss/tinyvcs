//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_FRAME_H_
#define TINYVCS_INCLUDE_FRAME_H_

#include <opencv2/cudaimgproc.hpp>
#include <fmt/ostream.h>
#include <fmt/format.h>

#include "infer.h"

struct Frame {
  Frame() = default;
  Frame(bool is_key, uint64_t pts, uint64_t dts, uint64_t index, const cv::cuda::GpuMat &image);
  void set_preprocess_image(const cv::cuda::GpuMat &image);
  void set_detections(const std::vector<Detection> &detections);
  std::string to_string();

  bool is_key_ = false;
  uint64_t pts_ = 0;
  uint64_t dts_ = 0;
  uint64_t index_ = 0;

  cv::cuda::GpuMat image_;
  cv::cuda::GpuMat preprocess_image_;
  std::vector<Detection> detections_;
};

#endif //TINYVCS_INCLUDE_FRAME_H_
