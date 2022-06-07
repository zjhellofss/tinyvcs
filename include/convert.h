//
// Created by fss on 22-6-7.
//

#ifndef TINYVCS_INCLUDE_YUV_CONVERT_H_
#define TINYVCS_INCLUDE_YUV_CONVERT_H_
struct AVFrame;
#include "opencv2/opencv.hpp"

bool Convert(AVFrame *frame, cv::Mat &image);

#endif //TINYVCS_INCLUDE_YUV_CONVERT_H_
