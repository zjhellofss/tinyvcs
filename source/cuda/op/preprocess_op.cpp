//
// Created by fss on 22-7-14.
//

#include "cuda/op/preprocess_op.h"

void PreProcessOp::ConvertYUV(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) const {
  output = cv::cuda::createContinuous(rows_, cols_, CV_8UC3);
  uchar *src = input.data;
  uchar *dst = output.data;
  int src_stride = input.cols;
  int dst_stride = input.cols * 3;
  convertYUV(src, rows_, cols_, src_stride, dst, dst_stride);
}

void PreProcessOp::Process(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) {
  CHECK(!input.empty());
  this->ConvertYUV(input, output);
  cv::cuda::resize(output, output, cv::Size(this->dst_cols_, this->dst_rows_));
}
