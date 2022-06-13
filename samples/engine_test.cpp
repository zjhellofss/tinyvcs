#include "opencv2/opencv.hpp"
#include "chain.h"

int main() {


  Inference infer("", "./tmp/v5.plan", 0, true);
  infer.Init();

  for (int b = 0; b < 2; ++b) {
    cv::Mat image1 = cv::imread("./tmp/bus.jpg");
    cv::Mat image2 = cv::imread("./tmp/bus.jpg");
    cv::Mat image3 = cv::imread("./tmp/zidane.jpg");
    cv::Mat image4 = cv::imread("./tmp/zidane.jpg");
    std::vector<cv::Mat> images;
    images.push_back(image1);
    images.push_back(image2);
    images.push_back(image3);
    images.push_back(image4);

    auto detections_all = infer.Infer(images, 0.2, 0.2);
    int i = 0;
    for (auto &image : images) {
      auto detections = detections_all.at(i);
      i += 1;
      for (int j = 0; j < detections.size(); ++j) {
        auto dect = detections.at(j);
        cv::rectangle(image, dect.box, cv::Scalar(255, 0, 0), 8);
      }
      cv::imwrite("./tmp/demo_" + std::to_string(i) + std::to_string(b) + ".jpg", image);
    }
  }
}