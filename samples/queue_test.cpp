//
// Created by fss on 22-6-16.
//
#include <benchmark/benchmark.h>
#include "opencv2/opencv.hpp"
#include "safevec.h"
#include "boost/lockfree/spsc_queue.hpp"

int main(){
  boost::lockfree::spsc_queue<cv::Mat, boost::lockfree::capacity<1024>> queue;
  queue.pop();

}