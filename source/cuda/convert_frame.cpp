//
// Created by fss on 22-7-1.
//
#include "cuda/convert_frame.h"
#include "opencv2/opencv.hpp"
#include "cuda_runtime_api.h"
#include "glog/logging.h"
#include "cuda/cuda_utils.h"
//ffmpeg
#include "ffmpeg.h"


std::optional<cv::cuda::GpuMat> ConvertFrame(AVFrame *cu_frame, cudaStream_t stream) {
  cv::cuda::GpuMat gpu_mat;
  if (cu_frame == nullptr) {
    LOG(ERROR) << "Frame is null";
    return std::nullopt;
  }
  AVPixelFormat hw_format = static_cast<AVPixelFormat>(cu_frame->format);
  if (hw_format != AVPixelFormat::AV_PIX_FMT_CUDA) {
    LOG(ERROR) << "Frame pixel format is not AV_PIX_FMT_CUDA";
    return std::nullopt;
  }

  cv::cuda::createContinuous(cu_frame->height * 3 / 2, cu_frame->width, CV_8UC1, gpu_mat);
  if (gpu_mat.empty()) {
    return std::nullopt;
  }

  int copy_offset = 0;
  if (!stream) CUDA_CHECK(cudaStreamCreate(&stream))

  for (size_t i = 0; i < FF_ARRAY_ELEMS(cu_frame->data) && cu_frame->data[i]; i++) {
    int src_pitch = cu_frame->linesize[i];
    int copy_width = cu_frame->width;
    int dst_pitch = copy_width;
    int copy_height = cu_frame->height >> ((i == 0 || i == 3) ? 0 : 1);
    CUDA_CHECK(cudaMemcpy2DAsync(gpu_mat.data + copy_offset,
                                 dst_pitch,
                                 cu_frame->data[i],
                                 src_pitch,
                                 copy_width,
                                 copy_height,
                                 cudaMemcpyDeviceToDevice, stream));
    copy_offset += copy_width * copy_height;
  }

  return gpu_mat;
}
