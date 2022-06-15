#include "opencv2/opencv.hpp"
#include "chain.h"

#include "tick.h"
#include "benchmark/benchmark.h"
std::vector<cv::Mat> images;
Inference infer("", "./tmp/v5m8.plan", 0, true);

// Define another benchmark
static void BM_Infer(benchmark::State &state) {
  auto detections_all = infer.Infer(images, 0.2, 0.2);
}
BENCHMARK(BM_Infer);

//BENCHMARK(BM_Infer)->Iterations(2000);
int main(int argc, char **argv) {
  infer.Init();
  cv::Mat image1 = cv::imread("./tmp/bus.jpg");
  for (int i = 0; i < 8; ++i) {
    images.push_back(image1);
  }

//  ::benchmark::Initialize(&argc, argv);
//  ::benchmark::RunSpecifiedBenchmarks();
  for (int i = 0; i < 16; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    TICK(INFER)
    auto detections_all = infer.Infer(images, 0.2, 0.2);
    TOCK_BATCH(INFER, 8)
  }
}