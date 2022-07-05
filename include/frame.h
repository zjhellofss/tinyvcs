//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_FRAME_H_
#define TINYVCS_INCLUDE_FRAME_H_



#include <opencv2/cudaimgproc.hpp>
#include "infer.h"

struct Frame {
  Frame(uint64_t pts, uint64_t origin_pts, cv::cuda::GpuMat image);
  Frame() = default;
  void set_preprocess_image(const cv::cuda::GpuMat &image);
  void set_detections(const std::vector<Detection> &detections);

  uint64_t pts_ = 0;
  uint64_t origin_pts_ = 0;
  cv::cuda::GpuMat image_;
  cv::cuda::GpuMat preprocess_image_;
  std::vector<Detection> detections_;
};
#endif //TINYVCS_INCLUDE_FRAME_H_
