//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_FRAME_H_
#define TINYVCS_INCLUDE_FRAME_H_

#include <cstdint>
#include <utility>
#include <memory>
struct AVFrame;
struct Frame {
 public:
  explicit Frame(int stream_idx, int64_t pts, std::shared_ptr<AVFrame> frame);
  std::shared_ptr<AVFrame> frame_ = nullptr;
  int stream_idx_ = -1; ///获取摄像机的编号
  int64_t pts_ = 0;
};
#endif //TINYVCS_INCLUDE_FRAME_H_
