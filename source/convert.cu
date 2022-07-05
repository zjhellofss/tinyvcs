//
// Created by fss on 22-6-7.
//
#include "convert.h"
#include <device_launch_parameters.h>
#include "cuda/preprocess.h"
#include "cuda_utils.h"
#include "ffmpeg.h"

#define NVXX1_CY 1220542
#define NVXX1_CUB 2116026
#define NVXX1_CUG -409993
#define NVXX1_CVG -852492
#define NVXX1_CVR 1673527
#define NVXX1_SHIFT 20

__host__ __device__
inline int divideUp(int total, int grain, int shift) {
  return (total + grain - 1) >> shift;
}

__device__ uchar saturateCast(int value) {
  unsigned int result = 0;
  asm("cvt.sat.u8.s32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
}

__device__  uchar3 convert2RGB_(const unsigned char &src_y, const uchar2 &src_uv) {
  int y = max(0, (src_y - 16)) * NVXX1_CY;
  int u = src_uv.x - 128;
  int v = src_uv.y - 128;

  int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
  int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
  int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

  int r = (y + ruv) >> NVXX1_SHIFT;
  int g = (y + guv) >> NVXX1_SHIFT;
  int b = (y + buv) >> NVXX1_SHIFT;

  uchar3 dst;
  dst.x = saturateCast(b);
  dst.y = saturateCast(g);
  dst.z = saturateCast(r);

  return dst;
}

__global__ void convert2RGB(const uchar *src,
                            int rows,
                            int cols,
                            size_t src_stride,
                            const uchar *dst,
                            size_t dst_stride) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar *input_y = (uchar *) src + element_y * src_stride;
  uchar2 *input_uv = (uchar2 *) ((uchar *) src +
      (rows + (element_y >> 1)) * src_stride);
  uchar3 *output = (uchar3 *) ((uchar *) dst + element_y * dst_stride);

  uchar value_y = input_y[element_x];
  uchar2 value_uv = input_uv[element_x >> 1];
  uchar3 result;
  result = convert2RGB_(value_y, value_uv);
  output[element_x] = result;

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = convert2RGB_(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = convert2RGB_(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = convert2RGB_(value_y, value_uv);
    output[element_x] = result;
  }
}

void convertFunction(const uchar *src, int rows, int cols, size_t src_stride, const uchar *dst, size_t dst_stride) {
  if (rows < 0 || cols < 0) {
    return;
  }
  if (!src || !dst) {
    return;
  }
  if (src_stride < cols || dst_stride < cols * 3) {
    return;
  }

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);
  convert2RGB<<<grid, block>>>(src, rows, cols, src_stride, dst, dst_stride);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
}