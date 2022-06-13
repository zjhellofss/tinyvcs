//#include "tensorrt/engine.h"
//
//#include "glog/logging.h"
//#include "opencv2/opencv.hpp"
//
//#include <memory>
//#include <cassert>
//#include <vector>
//
//typedef int i;
//typedef cv::Size size_;
//void sample_1() {
//  printf("--------sample 1-----------\n");
//  // create instance
//  std::unique_ptr<Trt> onnx_net{new Trt()};
//
//  // set dynamic shape config
//  onnx_net->AddDynamicShapeProfile("x", {4, 128, 128, 128}, {4, 128, 256, 256}, {4, 128, 512, 512});
//  onnx_net->AddDynamicShapeProfile("y", {4, 128, 128, 128}, {4, 128, 256, 256}, {4, 128, 512, 512});
//
//  // build engine
//  onnx_net->BuildEngine("./tmp/sample_add.onnx", "/tmp/v5.plan");
//
//  // do inference
//  int start = 128;
//  int step = 48;
//  for (int i = 0; i <= 8; i++) {
//    int h = start + i * step;
//    int w = start + i * step;
//    std::vector<float> x(4 * 128 * h * w, 1.0);
//    std::vector<float> y(4 * 128 * h * w, 2.0);
//    std::vector<float> z(4 * 128 * h * w, 0.0);
//
//    std::vector<int> shape{4, 128, h, w};
//    onnx_net->SetBindingDimensions(shape, 0);
//    onnx_net->SetBindingDimensions(shape, 1);
//
//    onnx_net->CopyFromHostToDevice(x, 0);
//    onnx_net->CopyFromHostToDevice(y, 1);
//    onnx_net->Forward();
//    onnx_net->CopyFromDeviceToHost(z, 2);
//
//    for (int j = 0; j < 4 * 128 * h * w; j++) {
//      assert(z[j] == 3.0);
//    }
//    printf("Test case with input shape 4x128x%dx%d PASSED\n", h, w);
//  }
//}
//
//void sample_2() {
//  printf("--------sample 2-----------\n");
//  // create instance
//  std::unique_ptr<Trt> onnx_net{new Trt()};
//
//  // set dynamic shape config
//  onnx_net->AddDynamicShapeProfile("x", {4, 128, 128, 128}, {4, 128, 256, 256}, {4, 128, 512, 512});
//  onnx_net->AddDynamicShapeProfile("y", {4, 128, 128, 128}, {4, 128, 256, 256}, {4, 128, 512, 512});
//
//  // build engine
//  onnx_net->DeserializeEngine("/tmp/v5.plan");
//
//  // do inference
//  int start = 128;
//  int step = 48;
//  for (int i = 0; i <= 8; i++) {
//    int h = start + i * step;
//    int w = start + i * step;
//    std::vector<float> x(4 * 128 * h * w, 1.0);
//    std::vector<float> y(4 * 128 * h * w, 2.0);
//    std::vector<float> z(4 * 128 * h * w, 0.0);
//
//    std::vector<int> shape{4, 128, h, w};
//    onnx_net->SetBindingDimensions(shape, 0);
//    onnx_net->SetBindingDimensions(shape, 1);
//
//    onnx_net->CopyFromHostToDevice(x, 0);
//    onnx_net->CopyFromHostToDevice(y, 1);
//    onnx_net->Forward();
//    onnx_net->CopyFromDeviceToHost(z, 2);
//
//    for (int j = 0; j < 4 * 128 * h * w; j++) {
//      assert(z[j] == 3.0);
//    }
//    printf("Test case with input shape 4x128x%dx%d PASSED\n", h, w);
//  }
//}
//
////void letterbox(const cv::Mat &image, cv::Mat &outImage,
////               const cv::Size &newShape = cv::Size(640, 640),
////               const cv::Scalar &color = cv::Scalar(114, 114, 114),
////               bool auto_ = false,
////               bool scaleFill = false,
////               bool scaleUp = true,
////               int stride = 32) {
////  cv::Size shape = image.size();
////  float r = std::min((float) newShape.height / (float) shape.height,
////                     (float) newShape.width / (float) shape.width);
////  if (!scaleUp)
////    r = std::min(r, 1.0f);
////
////  float ratio[2]{r, r};
////  int newUnpad[2]{(int) std::round((float) shape.width * r),
////                  (int) std::round((float) shape.height * r)};
////
////  auto dw = (float) (newShape.width - newUnpad[0]);
////  auto dh = (float) (newShape.height - newUnpad[1]);
////
////  if (auto_) {
////    dw = (float) ((int) dw % stride);
////    dh = (float) ((int) dh % stride);
////  } else if (scaleFill) {
////    dw = 0.0f;
////    dh = 0.0f;
////    newUnpad[0] = newShape.width;
////    newUnpad[1] = newShape.height;
////    ratio[0] = (float) newShape.width / (float) shape.width;
////    ratio[1] = (float) newShape.height / (float) shape.height;
////  }
////
////  dw /= 2.0f;
////  dh /= 2.0f;
////
////  if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
////    cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
////  }
////
////  int top = int(std::round(dh - 0.1f));
////  int bottom = int(std::round(dh + 0.1f));
////  int left = int(std::round(dw - 0.1f));
////  int right = int(std::round(dw + 0.1f));
////  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
////}
//
////size_t vectorProduct(const std::vector<int64_t> &vector) {
////  if (vector.empty())
////    return 0;
////
////  size_t product = 1;
////  for (const auto &element : vector)
////    product *= element;
////
////  return product;
////}
////struct Detection {
////  cv::Rect box;
////  float conf{};
////  int class_id{};
////};
//
//void scaleCoords(const cv::Size &imageShape, cv::Rect &coords, const cv::Size &imageOriginalShape) {
//  float gain = std::min((float) imageShape.height / (float) imageOriginalShape.height,
//                        (float) imageShape.width / (float) imageOriginalShape.width);
//
//  int pad[2] = {(int) (((float) imageShape.width - (float) imageOriginalShape.width * gain) / 2.0f),
//                (int) (((float) imageShape.height - (float) imageOriginalShape.height * gain) / 2.0f)};
//
//  coords.x = (int) std::round(((float) (coords.x - pad[0]) / gain));
//  coords.y = (int) std::round(((float) (coords.y - pad[1]) / gain));
//
//  coords.width = (int) std::round(((float) coords.width / gain));
//  coords.height = (int) std::round(((float) coords.height / gain));
//
//  // // clip coords, should be modified for width and height
//  // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
//  // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
//  // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
//  // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
//}
//
//void getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
//                      float &bestConf, int &bestClassId) {
//  // first 5 element are box and obj confidence
//  bestClassId = 5;
//  bestConf = 0;
//
//  for (int i = 5; i < numClasses + 5; i++) {
//    if (it[i] > bestConf) {
//      bestConf = it[i];
//      bestClassId = i - 5;
//    }
//  }
//}
//
//template<typename T>
//std::vector<T> flatten(const std::vector<std::vector<T>> &v) {
//  std::size_t total_size = 0;
//  for (const auto &sub : v)
//    total_size += sub.size(); // I wish there was a transform_accumulate
//  std::vector<T> result;
//  result.reserve(total_size);
//  for (const auto &sub : v)
//    result.insert(result.end(), sub.begin(), sub.end());
//  return result;
//}
//
//void sample_3() {
//  printf("--------sample 3-----------\n");
//  // create instance
//  std::unique_ptr<Trt> onnx_net{new Trt()};
//  SetDevice(0);
//  onnx_net->EnableFP16();
//  // build engine
////  onnx_net->BuildEngine("./tmp/yolov5s.onnx", "./tmp/v5.plan");
//  onnx_net->DeserializeEngine("./tmp/v5.plan");
//  int input_binding = onnx_net->GetNbInputBindings();
//  auto dims = onnx_net->GetBindingDims(input_binding);
//  for (int j = 0; j < dims.nbDims; j++) {
//    LOG(WARNING) << dims.d[j] << " x ";
//  }
//  int output_binding = onnx_net->GetNbOutputBindings();
//  dims = onnx_net->GetBindingDims(output_binding);
//  int batch = dims.d[0];
//  int channel = dims.d[1];
//  int height = dims.d[2];
//  int width = dims.d[3];
//
//  int num_classes = (int) dims.d[2] - 5;
//  int elements_in_batch = (int) (dims.d[1] * dims.d[2]);
//  LOG(WARNING) << "elements in batch: " << elements_in_batch;
//
//  int elements_in_all_batch = elements_in_batch * batch;
//  std::vector<std::vector<float>> input_tensor_values_all;
//  cv::Mat image;
//  for (int i = 0; i < batch; ++i) {
//    //read image
//    cv::Mat resized_image;
//    if (i == 2) {
//      image = cv::imread("./tmp/bus.jpg");
//    } else {
//      image = cv::imread("./tmp/zidane.jpg");
//    }
//    letterbox(image, resized_image);
//
//    cv::Mat float_image;
//    resized_image.convertTo(float_image, CV_32FC3, 1 / 255.0);
//    float *blob = new float[float_image.cols * float_image.rows * float_image.channels()];
//    cv::Size floatImageSize{float_image.cols, float_image.rows};
//
//    std::vector<cv::Mat> chw(float_image.channels());
//    for (int j = 0; j < float_image.channels(); ++j) {
//      chw[j] = cv::Mat(floatImageSize, CV_32FC1, blob + j * floatImageSize.width * floatImageSize.height);
//    }
//    cv::split(float_image, chw);
//
//    size_t input_tensor_size = vectorProduct({1, 3, 640, 640});
//    std::vector<float> input_tensor_values(blob, blob + input_tensor_size);
//    input_tensor_values_all.push_back(input_tensor_values);
//  }
//  std::vector<float> input;
//  input = flatten(input_tensor_values_all);
//  std::vector<float> output(elements_in_all_batch);
//  onnx_net->CopyFromHostToDevice(input, input_binding);
//  onnx_net->Forward();
//  onnx_net->CopyFromDeviceToHost(output, output_binding);
//  auto start = std::chrono::high_resolution_clock::now();
//  for (int i = 0; i < batch; ++i) {
//    std::vector<cv::Rect> boxes;
//    std::vector<float> confs;
//    std::vector<int> classIds;
//    auto begin = output.begin() + elements_in_batch * i;
//    for (auto it = begin; it != begin + elements_in_batch; it += num_classes + 5) {
//      float clsConf = it[4];
//
//      if (clsConf > 0.2) {
//        int centerX = (int) (it[0]);
//        int centerY = (int) (it[1]);
//        int width = (int) (it[2]);
//        int height = (int) (it[3]);
//        int left = centerX - width / 2;
//        int top = centerY - height / 2;
//
//        float objConf;
//        int classId;
//        getBestClassInfo(it, num_classes, objConf, classId);
//
//        float confidence = clsConf * objConf;
//
//        boxes.emplace_back(left, top, width, height);
//        confs.emplace_back(confidence);
//        classIds.emplace_back(classId);
//      }
//    }
//
//    std::vector<int> indices;
//    cv::dnn::NMSBoxes(boxes, confs, 0.2, 0.2, indices);
//    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;
//
//    std::vector<Detection> detections;
//    if(i==2){
//      image = cv::imread("./tmp/bus.jpg");
//    }
//    cv::Size resizedShape = cv::Size((int) 640, (int) 640);
//    for (int idx : indices) {
//      Detection det;
//      det.box = cv::Rect(boxes[idx]);
//      scaleCoords(resizedShape, det.box, image.size());
//
//      det.conf = confs[idx];
//      det.class_id = classIds[idx];
//      detections.emplace_back(det);
////      cv::rectangle(image, det.box, cv::Scalar(255, 0, 0), 8);
//    }
//  }
//  auto end = std::chrono::high_resolution_clock::now();
//  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//  std::cout << elapsed << std::endl;
//}
//
//int main(int argc, char *argv[]) {
//  google::InitGoogleLogging(argv[0]);
//  FLAGS_alsologtostderr = true;
//  sample_3();
//}
