//
// Created by fss on 22-6-6.
//
#include "frame.h"
Frame::Frame(int stream_idx, int64_t pts, std::shared_ptr<AVFrame> frame)
    : stream_idx_(stream_idx), pts_(pts), frame_(std::move(frame)) {

}
