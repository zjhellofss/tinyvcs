//
// Created by fss on 22-7-5.
//

#ifndef TINYVCS_INCLUDE_CUDA_PREPROCESS_H_
#define TINYVCS_INCLUDE_CUDA_PREPROCESS_H_
#include <opencv2/opencv.hpp>

enum DimX {
  kDimX0 = 16,
  kDimX1 = 32,
  kDimX2 = 32,
};

enum DimY {
  kDimY0 = 16,
  kDimY1 = 4,
  kDimY2 = 8,
};

enum ShiftX {
  kShiftX0 = 4,
  kShiftX1 = 5,
  kShiftX2 = 5,
};

enum ShiftY {
  kShiftY0 = 4,
  kShiftY1 = 2,
  kShiftY2 = 3,
};

enum BlockConfiguration0 {
  kBlockDimX0 = kDimX1,
  kBlockDimY0 = kDimY1,
  kBlockShiftX0 = kShiftX1,
  kBlockShiftY0 = kShiftY1,
};

std::shared_ptr<float> rgb2Plane(const float *src, int rows, int cols, int channels);
#endif //TINYVCS_INCLUDE_CUDA_PREPROCESS_H_
