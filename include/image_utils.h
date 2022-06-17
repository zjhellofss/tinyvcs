//
// Created by fss on 22-6-13.
//

#ifndef TINYVCS_INCLUDE_IMAGE_UTILS_H_
#define TINYVCS_INCLUDE_IMAGE_UTILS_H_
#include "opencv2/opencv.hpp"

void letterbox(const cv::Mat &image, cv::Mat &out_image,
               const cv::Size &new_shape = cv::Size(640, 640),
               const cv::Scalar &color = cv::Scalar(114, 114, 114),
               bool auto_ = false,
               bool scale_fill = false,
               bool scale_up = true,
               int stride = 32);

size_t vectorProduct(const std::vector<int64_t> &vector);

void scaleCoords(const cv::Size &image_shape, cv::Rect &coords, const cv::Size &image_original_shape);
#endif //TINYVCS_INCLUDE_IMAGE_UTILS_H_
