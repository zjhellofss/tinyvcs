//
// Created by fss on 22-7-14.
//

#ifndef TINYVCS_INCLUDE_CUDA_OP_PREPROCESS_OP_H_
#define TINYVCS_INCLUDE_CUDA_OP_PREPROCESS_OP_H_
#include "op.h"
#include "opencv2/cudawarping.hpp"
#include "ffmpeg.h"

#include "cuda/convert_yuv.h"

class PreProcessOp : public OP {
 public:
  explicit PreProcessOp(int rows, int cols, int dst_rows, int dst_cols)
      : rows_(rows), cols_(cols), dst_rows_(dst_rows), dst_cols_(dst_cols) {

  }

  void Process(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) override;
  void Process(cv::cuda::GpuMat &output) override {}
 private:
  void ConvertYUV(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) const;

  int rows_ = 0;
  int cols_ = 0;
  int dst_rows_ = 0;
  int dst_cols_ = 0;
};

#endif //TINYVCS_INCLUDE_CUDA_OP_PREPROCESS_OP_H_
