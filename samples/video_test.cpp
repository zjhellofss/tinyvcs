//
// Created by fss on 22-6-9.
//
#include "glog/logging.h"
#include "chain.h"

#include <vector>
#include <string>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
//  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  std::vector<std::string> subscriptions;
  subscriptions.emplace_back("ws://127.0.0.1:9002/");
  VideoStream stream(0, "rtsp://127.0.0.1:8554/mystream", subscriptions);
  bool b = stream.Open();
  stream.Run();

  return 0;
}