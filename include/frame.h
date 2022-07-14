//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_FRAME_H_
#define TINYVCS_INCLUDE_FRAME_H_

#include <vector>
#include "opencv2/cudaimgproc.hpp"
#include "fmt/ostream.h"
#include "fmt/format.h"

#include "infer.h"

struct Frame {
  Frame() = default;
  Frame(const cv::cuda::GpuMat &image,
        const cv::cuda::GpuMat &preprocess_image,
        int width,
        int height,
        bool is_key,
        uint64_t pts,
        uint64_t dts,
        uint64_t timestamp,
        uint64_t index);
  void set_detections(const std::vector<Detection> &detections);
  void set_cpu_image();
  void set_stream_id(int stream_id);
  std::string to_string();

  int stream_id_ = 0;
  int width_ = 0;
  int height_ = 0;
  bool is_key_ = false;
  uint64_t pts_ = 0;
  uint64_t dts_ = 0;
  uint64_t timestamp_ = 0;
  uint64_t index_ = 0;

  cv::Mat cpu_image_;
  std::vector<uchar> cpu_image_raw_;
  cv::cuda::GpuMat gpu_image_;
  cv::cuda::GpuMat preprocess_image_;
  std::vector<Detection> detections_;

  MSGPACK_DEFINE (index_, pts_, dts_, timestamp_, detections_, width_, height_, cpu_image_raw_, stream_id_, is_key_);
};

#endif //TINYVCS_INCLUDE_FRAME_H_
