//
// Created by fss on 22-7-14.
//

#ifndef TINYVCS_INCLUDE_CUDA_OP_OP_H_
#define TINYVCS_INCLUDE_CUDA_OP_OP_H_
#include "tensor.h"
class OP {
 public:
  virtual void Process(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) = 0;
  virtual void Process(cv::cuda::GpuMat &output) = 0;
};

#endif //TINYVCS_INCLUDE_CUDA_OP_OP_H_
