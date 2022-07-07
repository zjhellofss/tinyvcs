//
// Created by fss on 22-7-5.
//
#include "cuda/preprocess.h"
#include "cuda/cuda_utils.h"
#include <opencv2/opencv.hpp>

__global__ void rgb2PlanarKernel(const float *src, int rows, int cols, int channels,
                                 float *r, float *g, float *b) {
  extern __shared__ float3 s_data[];
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  if (idx > rows * cols) {
    return;
  }
  s_data[tid] = *reinterpret_cast<float3 *>(const_cast<float *>(src + idx * channels));
  __syncthreads();
  float3 pixel = s_data[tid];
  r[idx] = pixel.x;
  g[idx] = pixel.y;
  b[idx] = pixel.z;
}

std::shared_ptr<float> rgb2Planar(const float *src, int rows, int cols, int channels) {
  if (rows <= 0 || cols <= 0) {
    return nullptr;
  }
  if (channels != 3) {
    return nullptr;
  }
  float *dst = nullptr;
  cudaMalloc((void **) &dst, sizeof(float) * rows * cols * channels);
  std::shared_ptr<float> planar(dst, cudaFree);

  float *r = dst;
  float *g = dst + 1 * rows * cols;
  float *b = dst + 2 * rows * cols;

  int threads = kDimX1 << kShiftX1;
  int blocks = (rows * cols + threads - 1) / threads;
  rgb2PlanarKernel<<<blocks, threads, sizeof(float3) * threads>>>(src, rows, cols, channels, r, g, b);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError())
  return planar;
}