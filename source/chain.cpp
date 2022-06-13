#include "chain.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "fmt/core.h"
#include "websocket/client.h"
#include "json/jsonbuilder.h"
#include "glog/logging.h"

#include "player.h"
#include "tensorrt/engine.h"
#include "image_utils.h"

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &v) {
  std::size_t total_size = 0;
  for (const auto &sub : v)
    total_size += sub.size(); // I wish there was a transform_accumulate
  std::vector<T> result;
  result.reserve(total_size);
  for (const auto &sub : v)
    result.insert(result.end(), sub.begin(), sub.end());
  return result;
}

void getBestClassInfo(std::vector<float>::iterator it, const int &num_classes,
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

void VideoStream::Run() {
  std::thread t([this]() {
    this->ReadImages();
  });
  threads_.push_back(std::move(t));
}

bool VideoStream::Open() {
  player_ = std::make_shared<Player>(stream_id_, rtsp_address_);
  bool open_success = player_->Open();
  // open websockets
  for (const std::string &subscription : subscriptions_) {
    std::shared_ptr<Connection> connection = std::make_shared<Connection>();
    bool is_open = connection->Connect(subscription);
    if (!is_open) {
      LOG(FATAL) << "Can not connect to " << subscription;
      return false;
    } else {
      connnections_.push_back(connection);
    }
  }
  if (!open_success) {
    return false;
  }
  player_->Run();
  return true;
}
void VideoStream::ReadImages() {
  while (true) {
    try {
      Frame frame = player_->GetImage();
      cv::Mat image = frame.image_;
      if (!image.empty()) {
        int height = image.size().height;
        int width = image.size().width;
        vmaps values;
        values.insert({"pts", frame.pts_});
        values.insert({"width", width});
        values.insert({"height", height});
        std::string json_res = create_json(values);
        for (const auto &connection : this->connnections_) {
          if (connection->IsRunnable()) {
            bool send_success = connection->Send(json_res);
            if (send_success) {
            }
          }
        }
      }
    }
    catch (SynchronizedVectorException &e) {
      if (!player_->IsRunnable()) {
        LOG(ERROR) << "Decode packet process is exited with error";
        break;
      }
    }
  }
  LOG(INFO) << "Read images process is exited!";
}

void Inference::Init() {
  this->onnx_net_ = std::make_unique<Trt>();
  SetDevice(device_);
  if (this->enable_fp16_)
    this->onnx_net_->EnableFP16();
  LOG_IF(FATAL, this->engine_file_.empty()) << "engine file is empty";

  if (this->engine_file_.empty()) {
    this->onnx_net_->BuildEngine(this->onnx_file_, this->engine_file_);
    LOG(INFO) << "build engine successfully!";
  } else {
    bool ret = this->onnx_net_->DeserializeEngine(this->engine_file_);
    LOG_IF(FATAL, ret != true) << "deserialize engine failed";
  }
  input_binding_ = onnx_net_->GetNbInputBindings();
  input_dims_ = onnx_net_->GetBindingDims(input_binding_);
  for (int j = 0; j < input_dims_.nbDims; j++) {
    LOG(INFO) << input_dims_.d[j] << " x ";
  }

  output_binding_ = onnx_net_->GetNbOutputBindings();
  output_dims_ = onnx_net_->GetBindingDims(output_binding_);
  batch_ = output_dims_.d[0];
  LOG_IF(FATAL, batch_ < 0) << "inference batch is less than zero";

  num_classes_ = (int) output_dims_.d[2] - 5;
  elements_in_one_batch_ = (int) (output_dims_.d[1] * output_dims_.d[2]);
  LOG(INFO) << "elements in batch: " << elements_in_one_batch_;

  elements_in_all_batch_ = elements_in_one_batch_ * batch_;
}

std::vector<std::vector<Detection>> Inference::Infer(const std::vector<cv::Mat> &images,
                                                     float conf_thresh,
                                                     float iou_thresh) {
  std::vector<std::vector<Detection>> detections;
  if (images.size() != batch_) {
    LOG(ERROR) << "infer images not equal to batch";
    return detections;
  }
  std::vector<std::vector<float>> input_tensor_values_all;
  for (int i = 0; i < batch_; ++i) {
    cv::Mat resized_image;
    const cv::Mat &image = images.at(i);
    LOG_IF(FATAL, image.empty()) << "has empty image";
    letterbox(image, resized_image);

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC3, 1 / 255.0);
    std::shared_ptr<float>
        blob = std::shared_ptr<float>(new float[float_image.cols * float_image.rows * float_image.channels()]);
    cv::Size float_image_size{float_image.cols, float_image.rows};

    std::vector<cv::Mat> chw(float_image.channels());
    for (int j = 0; j < float_image.channels(); ++j) {
      chw[j] = cv::Mat(float_image_size, CV_32FC1, blob.get() + j * float_image_size.width * float_image_size.height);
    }
    cv::split(float_image, chw);

    size_t input_tensor_size = vectorProduct({1, input_dims_.d[1], input_dims_.d[2], input_dims_.d[3]});
    std::vector<float> input_tensor_values(blob.get(), blob.get() + input_tensor_size);
    input_tensor_values_all.push_back(input_tensor_values);
  }

  std::vector<float> input;
  input = flatten(input_tensor_values_all);
  std::vector<float> output(elements_in_all_batch_);
  onnx_net_->CopyFromHostToDevice(input, input_binding_);
  onnx_net_->Forward();
  onnx_net_->CopyFromDeviceToHost(output, output_binding_);

  for (int i = 0; i < batch_; ++i) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
    auto begin = output.begin() + elements_in_one_batch_ * i;
    for (auto it = begin; it != begin + elements_in_one_batch_; it += num_classes_ + 5) {
      float cls_conf = it[4];

      if (cls_conf > 0.2) {
        int centerX = (int) (it[0]);
        int centerY = (int) (it[1]);
        int width = (int) (it[2]);
        int height = (int) (it[3]);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

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
      scaleCoords(resized_shape, det.box, images.at(i).size());

      det.conf = confs[idx];
      det.class_id = class_ids[idx];
      detections_batch.emplace_back(det);
    }
    detections.push_back(detections_batch);
  }
  return detections;
}


