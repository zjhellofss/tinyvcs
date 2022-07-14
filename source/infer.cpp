//
// Created by fss on 22-6-17.
//

#include "infer.h"
#include "glog/logging.h"
#include "tick.h"
#include "image_utils.h"
#include "cuda/preprocess.h"
#include "cuda/post_processing.cuh"

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
  if ((int) images.size() != batch_) {
    LOG(ERROR) << "infer images not equal to batch";
    return detections;
  }
  auto start = std::chrono::steady_clock::now();

  size_t input_tensor_size = vectorProduct({1, input_dims_.d[1], input_dims_.d[2], input_dims_.d[3]});
  float *input_raw = nullptr;
  cudaMalloc((void **) &input_raw, sizeof(float) * input_tensor_size * batch_);
  std::shared_ptr<float> input = std::shared_ptr<float>(input_raw, cudaFree);

  for (int i = 0; i < batch_; ++i) {
    cv::cuda::GpuMat image_gpu = images.at(i);

    if (image_gpu.empty() || image_gpu.size().width != width || image_gpu.size().height != height
        || image_gpu.channels() != 3) {
      LOG(ERROR) << "has wrong gpu image";
      return detections;
    }

    cv::Mat image_output;
    image_gpu.convertTo(image_gpu, CV_32FC3, 1 / 255.);
    std::shared_ptr<float> data_raw = rgb2Plane(reinterpret_cast<float *>(image_gpu.data), height, width, 3);

    cudaMemcpy(input.get() + i * input_tensor_size,
               data_raw.get(),
               input_tensor_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }

  float *output_raw = nullptr;
  cudaMalloc(&output_raw, sizeof(float) * elements_in_all_batch_);
  std::shared_ptr<float> output(output_raw, [](float *ptr) {
    if (ptr) {
      cudaFree(ptr);
    }
  });
  onnx_net_->CopyFromDeviceToDevice(input.get(), input_tensor_size * batch_, input_binding_);
  onnx_net_->Forward();
  onnx_net_->CopyFromDeviceToDevice2(output_raw, elements_in_all_batch_, output_binding_);

  size_t detections_sizes = 0;
  int max_det_count = (elements_in_one_batch_) / (num_classes_ + 5);
  Detection_ *detections_raw_ = nullptr;

  for (int i = 0; i < batch_; ++i) {
    auto inputs = output_raw + elements_in_one_batch_ * i;
    cudaMallocManaged((void **) &detections_raw_, sizeof(Detection_) * max_det_count);

    std::shared_ptr<Detection_>
        detections_ = std::shared_ptr<Detection_>(detections_raw_, [](Detection_ *detections_raw_) {
      if (detections_raw_) {
        cudaFree(detections_raw_);
      }
    });

    std::vector<Detection_> results =
        postProcessing(inputs, conf_thresh, num_classes_, max_det_count, detections_raw_);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    for (auto detection_ : results) {
      confidences.push_back(detection_.obj_conf_);
      class_ids.push_back(detection_.class_id_);
      boxes.emplace_back(detection_.left_, detection_.top_, detection_.box_width_, detection_.box_height_);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, iou_thresh, indices);
    cv::Size resized_shape = cv::Size((int) input_dims_.d[2], (int) input_dims_.d[3]);
    std::vector<Detection> detections_batch;

    for (int idx : indices) {
      Detection det;
      det.box = cv::Rect(boxes[idx]);
      scaleCoords(resized_shape, det.box, cv::Size{960, 640});

      det.conf = confidences[idx];
      det.class_id = class_ids[idx];
      detections_batch.emplace_back(det);
    }
    detections_sizes += detections_batch.size();
    detections.push_back(detections_batch);
  }
  auto end = std::chrono::steady_clock::now();
  long infer_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / batch_;
  this->infer_costs_.push_back(infer_cost);
//  LOG(INFO) << "infer cost: " <<
//            << " ms"
//            << " infer number: " << detections_sizes;

  return detections;
}

float Inference::infer_costs() {
  if (this->infer_costs_.empty()) {
    return 0.f;
  }
  long sum = std::accumulate(this->infer_costs_.begin(), this->infer_costs_.end(), 0l);
  float mean = (float) sum / (float) this->infer_costs_.size();
  return mean;
}
