#include "opencv2/opencv.hpp"
#include "chain.h"

#include "benchmark/benchmark.h"
std::vector<cv::Mat> images;
Inference infer("", "./tmp/v5.plan", 0, true);

// Define another benchmark
static void BM_Infer(benchmark::State &state) {
  auto detections_all = infer.Infer(images, 0.2, 0.2);
}
BENCHMARK(BM_Infer);

//BENCHMARK(BM_Infer)->Iterations(2000);
int main(int argc, char **argv) {
  infer.Init();
  cv::Mat image1 = cv::imread("./tmp/bus.jpg");
  cv::Mat image2 = cv::imread("./tmp/bus.jpg");
  cv::Mat image3 = cv::imread("./tmp/zidane.jpg");
  cv::Mat image4 = cv::imread("./tmp/zidane.jpg");
  images.push_back(image1);
  images.push_back(image2);
  images.push_back(image3);
  images.push_back(image4);

//  ::benchmark::Initialize(&argc, argv);
//  ::benchmark::RunSpecifiedBenchmarks();
  for (int i = 0; i < 16; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto detections_all = infer.Infer(images, 0.2, 0.2);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("%ld ms\n", elapsed);
  }
}