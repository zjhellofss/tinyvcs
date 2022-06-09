//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_FRAME_H_
#define TINYVCS_INCLUDE_FRAME_H_

#include <opencv2/opencv.hpp>
struct Frame {
  Frame(int pts, int origin_pts, cv::Mat image);
  int pts_ = 0;
  int origin_pts_ = 0;
  cv::Mat image_;
};
#endif //TINYVCS_INCLUDE_FRAME_H_
