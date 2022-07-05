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

std::vector<std::vector<Detection>> Inference::Infer(const std::vector<cv::Mat> &images,
                                                     float conf_thresh,
                                                     float iou_thresh) {
  TICK(ALL)
  std::vector<std::vector<Detection>> detections;
  if (images.size() != batch_) {
    LOG(ERROR) << "infer images not equal to batch";
    return detections;
  }
  size_t input_tensor_size = vectorProduct({1, input_dims_.d[1], input_dims_.d[2], input_dims_.d[3]});
  float *input_raw = nullptr;
  cudaMalloc((void **) &input_raw, sizeof(float) * input_tensor_size * batch_);
  std::shared_ptr<float> input = std::shared_ptr<float>(input_raw, cudaFree);

  int cols = images.at(0).cols;
  int rows = images.at(0).rows;
  int channels = images.at(0).channels();
  if (!blob_) {
    blob_ = std::unique_ptr<float>(new float[rows * cols * channels]);
  }
  std::vector<cv::Mat> chw(channels);

  for (size_t i = 0; i < batch_; ++i) {
    cv::Mat image = images.at(i);

    if (image.rows != rows || image.cols != cols || image.channels() != channels) {
      LOG(FATAL) << "do not have a same size";
    }
    LOG_IF(FATAL, image.empty()) << "has empty image";

    cv::cuda::GpuMat image_gpu;
    image_gpu.upload(image);

    image_gpu.convertTo(image_gpu, CV_32FC3, 1 / 255.);
    std::shared_ptr<float> data_raw = rgb2Planar(reinterpret_cast<float *>(image_gpu.data), rows, cols, channels);

    cudaMemcpy(input.get() + i * input_tensor_size,
                    data_raw.get(),
                    input_tensor_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
  }

  std::vector<float> output(elements_in_all_batch_);
  onnx_net_->CopyFromDeviceToDevice(input.get(), input_tensor_size * batch_, input_binding_);
  onnx_net_->Forward();
  onnx_net_->CopyFromDeviceToHost(output, output_binding_);

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
        int width = (int) (it[2]);
        int height = (int) (it[3]);
        int left = center_x - width / 2;
        int top = center_y - height / 2;

        float obj_conf;
        int class_id;
        getBestClassInfo(it, num_classes_, obj_conf, class_id);

        float confidence = cls_conf * obj_conf;

        boxes.emplace_back(left, top, width, height);
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
    detections.push_back(detections_batch);
  }
  TOCK_BATCH(ALL, batch_)
  return detections;
}
