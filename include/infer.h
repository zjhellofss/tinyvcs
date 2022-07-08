//
// Created by fss on 22-6-17.
//

#ifndef TINYVCS_SOURCE_INFER_H_
#define TINYVCS_SOURCE_INFER_H_
#include <string>

#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "boost/core/noncopyable.hpp"
#include "NvInferVersion.h"

#include "nlohmann/json.hpp"
#include "tensorrt/engine.h"

using json = nlohmann::json;

struct Detection {
  cv::Rect box;
  int class_id = 0;
  float conf = 0.f;

  void to_json(json &j) const {
    j = json{{"x", this->box.x},
             {"y", this->box.y},
             {"height", this->box.height},
             {"width", this->box.width},
             {"class_id", this->class_id},
             {"conf", this->conf}
    };
  }
};

class Inference : private boost::noncopyable {
 public:
  Inference(std::string onnx_file, std::string engine_file, int device, bool enable_fp16)
      : onnx_file_(std::move(onnx_file)),
        engine_file_(std::move(engine_file)),
        enable_fp16_(enable_fp16),
        device_(device) {

  }
  void Init();

  std::vector<std::vector<Detection>> Infer(const std::vector<cv::cuda::GpuMat> &images,
                                            int width,
                                            int height,
                                            float conf_thresh,
                                            float iou_thresh);

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
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims output_dims_{};
};

#endif //TINYVCS_SOURCE_INFER_H_
