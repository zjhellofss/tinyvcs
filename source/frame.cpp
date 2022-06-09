//
// Created by fss on 22-6-6.
//
#include "frame.h"
#include <utility>
Frame::Frame(int pts, int origin_pts, cv::Mat image) : pts_(pts), origin_pts_(origin_pts), image_(std::move(image)) {}
