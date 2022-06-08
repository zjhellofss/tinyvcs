//
// Created by fss on 22-6-7.
//

#include "convert.h"
#include "glog/logging.h"
#include "ffmpeg.h"
#include "libyuv/convert.h"

bool Convert(AVFrame *frame, cv::Mat &image) {

  if (frame == nullptr || image.empty()) {
    LOG(ERROR) << "frame or image is empty";
    return false;
  }
  if (image.size().width != frame->width ||
      image.size().height != frame->height || image.type() != CV_8UC4) {
    LOG(ERROR) << "Input opencv mat do not have correct sizes";
  }
  int ret = -1;
  switch (frame->format) {
    case AVPixelFormat::AV_PIX_FMT_YUV420P: {
      ret = libyuv::I420ToARGB(frame->data[0], frame->linesize[0],
                               frame->data[1], frame->linesize[1],
                               frame->data[2], frame->linesize[2],
                               image.data, frame->width * 4,
                               frame->width,
                               frame->height);

      break;
    }
    case AVPixelFormat::AV_PIX_FMT_YUVJ420P: {
      ret = libyuv::J420ToARGB(frame->data[0], frame->linesize[0],
                               frame->data[1], frame->linesize[1],
                               frame->data[2], frame->linesize[2],
                               image.data, frame->width * 4,
                               frame->width,
                               frame->height);
      break;
    }
    case AVPixelFormat::AV_PIX_FMT_NV12: {
      ret = libyuv::NV12ToARGB(frame->data[0],
                               frame->linesize[0],
                               frame->data[1],
                               frame->linesize[1],
                               image.data,
                               frame->width * 4,
                               frame->width,
                               frame->height);
      break;
    }
    default: {
      LOG(WARNING) << "Frame pixel format " << frame->format << " is not supported!";
      return false;
    }
  }
  if (!ret) {
    return true;
  } else {
    LOG(WARNING) << "Convert pixel format failed!";
    return false;
  }
}