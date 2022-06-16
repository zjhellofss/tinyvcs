//
// Created by fss on 22-6-6.
//
#include "frame.h"
#include <utility>
Frame::Frame(uint64_t pts, uint64_t origin_pts, cv::Mat image) : pts_(pts), origin_pts_(origin_pts), image_(std::move(image)) {}
