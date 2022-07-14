//
// Created by fss on 22-7-14.
//

#ifndef TINYVCS_INCLUDE_CUDA_OP_AVFRAME_OP_H_
#define TINYVCS_INCLUDE_CUDA_OP_AVFRAME_OP_H_
#include "op.h"
#include "ffmpeg.h"
#include "cuda/convert_frame.h"

class FrameConvertOp : public OP {

 public:
  FrameConvertOp(const std::shared_ptr<AVFrame> &frame) : frame_(frame) {
  }

  void Process(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) override {

  }

  void Process(cv::cuda::GpuMat &output) override {
    CHECK(frame_);
    if (!stream_) {
      cudaStreamCreate(&stream_);
    }
    std::optional<cv::cuda::GpuMat> image_opt = ConvertFrame(frame_.get(), this->stream_);
    if (image_opt.has_value()) {
      cv::cuda::GpuMat image = image_opt.value();
      output = image;
    }
  }

  virtual ~FrameConvertOp() {
    if (stream_)
      cudaStreamDestroy(stream_);
  }
 private:
  cudaStream_t stream_ = nullptr;
  std::shared_ptr<AVFrame> frame_;
};
#endif //TINYVCS_INCLUDE_CUDA_OP_AVFRAME_OP_H_
