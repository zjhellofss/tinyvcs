//
// Created by fss on 22-7-11.
//

#ifndef TINYVCS_INCLUDE_CUDA_POST_PROCESSING_CUH_
#define TINYVCS_INCLUDE_CUDA_POST_PROCESSING_CUH_
#include <vector>
struct Detection_ {
  float obj_conf_ = 0.f;
  int box_width_ = 0;
  int box_height_ = 0;
  int left_ = 0;
  int top_ = 0;
  int class_id_ = 0;
  bool is_valid_ = false;

  __device__ void set_info(float obj_conf,
                           int box_width,
                           int box_height,
                           int left,
                           int top,
                           int class_id) {
    obj_conf_ = obj_conf;
    box_width_ = box_width;
    box_height_ = box_height;
    left_ = left;
    top_ = top;
    class_id_ = class_id;
    is_valid_ = true;
  }

  __device__ void set_invalid() {
    is_valid_ = false;
  }
};

std::vector<Detection_> postProcessing(float *inputs,
                                       float conf_thresh,
                                       int num_classes,
                                       int elements_number,
                                       Detection_ *detections);

#endif //TINYVCS_INCLUDE_CUDA_POST_PROCESSING_CUH_
