//
// Created by fss on 22-6-9.
//

#ifndef TINYVCS_INCLUDE_CHAIN_H_
#define TINYVCS_INCLUDE_CHAIN_H_
#include <string>
#include <vector>
#include <thread>

#include "websocket/client.h"

#include "tensorrt/engine.h"
#include "player.h"

struct Detection {
  cv::Rect box;
  float conf{};
  int class_id{};
};

class Inference {
 public:
  Inference(std::string onnx_file, std::string engine_file, int device, bool enable_fp16)
      : onnx_file_(std::move(onnx_file)),
        engine_file_(std::move(engine_file)),
        enable_fp16_(enable_fp16),
        device_(device) {

  }
  void Init();

  std::vector<std::vector<Detection>> Infer(const std::vector<cv::Mat> &images, float conf_thresh, float iou_thresh);

 private:
  std::unique_ptr<Trt> onnx_net_;
  std::string onnx_file_;
  std::string engine_file_;
  bool enable_fp16_ = false;
  int device_ = 0;
  size_t batch_ = 0;
  int num_classes_ = 0;
  int elements_in_one_batch_ = 0;
  int elements_in_all_batch_ = 0;
  int output_binding_ = 0;
  int input_binding_ = 0;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
};

class VideoStream {
 public:
  explicit VideoStream(int stream_id,
                       int duration,
                       std::string rtsp_address,
                       std::vector<std::string> subscriptions)
      : stream_id_(stream_id),
        duration_(duration),
        rtsp_address_(std::move(rtsp_address)),
        subscriptions_(std::move(subscriptions)) {

  }
  ~VideoStream() {
    for (std::thread &t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  bool Open();

  void ReadImages();

  void Infer();

  void set_inference(size_t batch, const std::string &engine_file);

  void Run();

 private:
  int stream_id_ = 0;
  size_t batch_ = 0;
  int duration_ = 0;
  std::string rtsp_address_;
  std::vector<std::string> subscriptions_;
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<Connection>> connnections_;
  boost::lockfree::spsc_queue<Frame, boost::lockfree::capacity<1024>> frames_;
  std::shared_ptr<Player> player_;
  std::unique_ptr<Inference> inference_;
};

#endif //TINYVCS_INCLUDE_CHAIN_H_
