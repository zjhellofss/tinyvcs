//
// Created by fss on 22-7-14.
//

#ifndef TINYVCS_SOURCE_CUDA_OP_LETTERBOX_OP_H_
#define TINYVCS_SOURCE_CUDA_OP_LETTERBOX_OP_H_
#include "cuda/op/op.h"
#include "image_utils.h"

class LetterBoxOp : public OP {
 public:
  LetterBoxOp(int dst_rows, int dst_cols) : dst_rows_(dst_rows), dst_cols_(dst_cols) {

  }
  void Process(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) override {
    letterbox(input, output);
  }
  void Process(cv::cuda::GpuMat &output) override {}

 private:
  int dst_rows_;
  int dst_cols_;
};
#endif //TINYVCS_SOURCE_CUDA_OP_LETTERBOX_OP_H_
