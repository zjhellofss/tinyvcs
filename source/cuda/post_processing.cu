//
// Created by fss on 22-7-11.
//
#include "cuda/post_processing.cuh"

#include <memory>
#include "cuda/cuda_utils.h"
#include "glog/logging.h"

__device__ int getBestClassInfo(const float *elements, const int &num_classes,
                                float &best_conf, int best_class_id) {

  best_class_id = 5;
  best_conf = 0;
  for (int i = 5; i < num_classes + 5; i++) {
    if (elements[i] > best_conf) {
      best_conf = elements[i];
      best_class_id = i - 5;
    }
  }
  return best_class_id;
}

static __global__ void postProcessingCu(float *inputs,
                                        float conf_thresh,
                                        int num_classes,
                                        int elements_number,
                                        Detection_ *detections) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > elements_number) {
    return;
  }
  auto elements = inputs + tid * (num_classes + 5);
  float cls_conf = elements[4];

  if (cls_conf < conf_thresh) {
    return;
  }
  int center_x = (int) (elements[0]);
  int center_y = (int) (elements[1]);
  int box_width = (int) (elements[2]);
  int box_height = (int) (elements[3]);
  int left = center_x - box_width / 2;
  int top = center_y - box_height / 2;

  float obj_conf = 0.f;
  int class_id = 0;
  getBestClassInfo(elements, num_classes, obj_conf, class_id);

  detections[tid].set_info(obj_conf, box_width, box_height, left, top, class_id);
}

std::vector<Detection_> postProcessing(float *inputs,
                                       float conf_thresh,
                                       int num_classes,
                                       int elements_number,
                                       Detection_ *detections) {
  int threads = 1024;
  int blocks = (elements_number + threads - 1) / threads;

  postProcessingCu<<<blocks, threads>>>(inputs,
                                        conf_thresh,
                                        num_classes,
                                        elements_number,
                                        detections);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError())
  std::vector<Detection_> results;
  for (int i = 0; i < elements_number; ++i) {
    if (detections[i].is_valid_) {
      results.push_back(detections[i]);
    }
  }
  return results;
}
