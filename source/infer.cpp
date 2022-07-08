//
// Created by fss on 22-6-17.
//

#include "infer.h"
#include "glog/logging.h"
#include "tick.h"
#include "image_utils.h"
#include "cuda/preprocess.h"

static void getBestClassInfo(std::vector<float>::iterator it, const int &num_classes,
                             float &best_conf, int &best_class_id) {
  // first 5 element are box and obj confidence
  best_class_id = 5;
  best_conf = 0;

  for (int i = 5; i < num_classes + 5; i++) {
    if (it[i] > best_conf) {
      best_conf = it[i];
      best_class_id = i - 5;
    }
  }
}

void Inference::Init() {
  this->onnx_net_ = std::make_unique<Trt>();
  SetDevice(device_);
  if (this->enable_fp16_)
    this->onnx_net_->EnableFP16();
  LOG_IF(FATAL, this->engine_file_.empty()) << "engine file is empty";

  bool ret = this->onnx_net_->DeserializeEngine(this->engine_file_);
  LOG_IF(FATAL, ret != true) << "deserialize engine failed";
  input_binding_ = onnx_net_->input_bindings();
  input_dims_ = onnx_net_->binding_dims(input_binding_);
  for (int j = 0; j < input_dims_.nbDims; j++) {
    LOG(INFO) << input_dims_.d[j] << " x ";
  }

  output_binding_ = onnx_net_->output_bindings();
  output_dims_ = onnx_net_->binding_dims(output_binding_);
  batch_ = output_dims_.d[0];
  LOG_IF(FATAL, batch_ == 0) << "inference batch is less than zero";

  num_classes_ = (int) output_dims_.d[2] - 5;
  elements_in_one_batch_ = (int) (output_dims_.d[1] * output_dims_.d[2]);
  LOG(INFO) << "elements in batch: " << elements_in_one_batch_;

  elements_in_all_batch_ = elements_in_one_batch_ * batch_;
}

std::vector<std::vector<Detection>> Inference::Infer(const std::vector<cv::cuda::GpuMat> &images,
                                                     int width,
                                                     int height,
                                                     float conf_thresh,
                                                     float iou_thresh) {
  std::vector<std::vector<Detection>> detections;
  if (images.size() != batch_) {
    LOG(ERROR) << "infer images not equal to batch";
    return detections;
  }
  auto start = std::chrono::steady_clock::now();

  size_t input_tensor_size = vectorProduct({1, input_dims_.d[1], input_dims_.d[2], input_dims_.d[3]});
  float *input_raw = nullptr;
  cudaMalloc((void **) &input_raw, sizeof(float) * input_tensor_size * batch_);
  std::shared_ptr<float> input = std::shared_ptr<float>(input_raw, cudaFree);

//  int cols = images.at(0).cols;
//  int rows = images.at(0).rows;
//  int channels = images.at(0).channels();

  for (size_t i = 0; i < batch_; ++i) {
    cv::cuda::GpuMat image_gpu = images.at(i);

    if (image_gpu.empty() || image_gpu.size().width != width || image_gpu.size().height != height
        || image_gpu.channels() != 3) {
      LOG(ERROR) << "has wrong gpu image";
      return detections;
    }

    cv::Mat image_output;
    image_gpu.download(image_output);
    image_gpu.convertTo(image_gpu, CV_32FC3, 1 / 255.);
    std::shared_ptr<float> data_raw = rgb2Plane(reinterpret_cast<float *>(image_gpu.data), height, width, 3);

    cudaMemcpy(input.get() + i * input_tensor_size,
               data_raw.get(),
               input_tensor_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }

  std::vector<float> output(elements_in_all_batch_);
  onnx_net_->CopyFromDeviceToDevice(input.get(), input_tensor_size * batch_, input_binding_);
  onnx_net_->Forward();
  onnx_net_->CopyFromDeviceToHost(output, output_binding_);

  size_t detections_sizes = 0;
  for (size_t i = 0; i < batch_; ++i) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
    auto begin = output.begin() + elements_in_one_batch_ * i;
    for (auto it = begin; it != begin + elements_in_one_batch_; it += num_classes_ + 5) {
      float cls_conf = it[4];

      if (cls_conf > conf_thresh) {
        int center_x = (int) (it[0]);
        int center_y = (int) (it[1]);
        int box_width = (int) (it[2]);
        int box_height = (int) (it[3]);
        int left = center_x - box_width / 2;
        int top = center_y - box_height / 2;

        float obj_conf;
        int class_id;
        getBestClassInfo(it, num_classes_, obj_conf, class_id);
        float confidence = cls_conf * obj_conf;

        boxes.emplace_back(left, top, box_width, box_height);
        confs.emplace_back(confidence);
        class_ids.emplace_back(class_id);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);
    cv::Size resized_shape = cv::Size((int) input_dims_.d[2], (int) input_dims_.d[3]);
    std::vector<Detection> detections_batch;
    for (int idx : indices) {
      Detection det;
      det.box = cv::Rect(boxes[idx]);
      scaleCoords(resized_shape, det.box, cv::Size{960, 640});

      det.conf = confs[idx];
      det.class_id = class_ids[idx];
      detections_batch.emplace_back(det);
    }
    detections_sizes += detections_batch.size();
    detections.push_back(detections_batch);
  }
  auto end = std::chrono::steady_clock::now();
  LOG(INFO) << "infer cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / batch_
            << " ms"
            << " infer number: " << detections_sizes;

  return detections;
}
