//
// Created by fss on 22-6-7.
//

#ifndef TINYVCS_INCLUDE_YUV_CONVERT_H_
#define TINYVCS_INCLUDE_YUV_CONVERT_H_
struct AVFrame;
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"

void convertYUV(const uchar *src, int rows, int cols, size_t src_stride, const uchar *dst, size_t dst_stride);
#endif //TINYVCS_INCLUDE_YUV_CONVERT_H_
