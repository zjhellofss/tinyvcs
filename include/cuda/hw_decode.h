
#ifndef TINYVCS_INCLUDE_HWDECODE_H_
#define TINYVCS_INCLUDE_HWDECODE_H_
#include <optional>
#include "opencv2/opencv.hpp"
struct AVFrame;
std::optional<cv::cuda::GpuMat> ConvertFrame(AVFrame *cu_frame);
#endif