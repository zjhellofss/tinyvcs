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

  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    std::thread t([&, i] {
      VideoStream stream(i + 3,
                         3,
                         "rtsp://127.0.0.1:8554/mystream",
                         subscriptions);
      bool b = stream.Open();
      stream.set_inference(8,"/home/fss/code/origin_vsc/tinyvcs/tmp/v5m8.plan");
      assert(b);
      stream.Run();
    });
    threads.push_back(std::move(t));
  }

  for (size_t i = 0; i < threads.size(); ++i) {
    if (threads.at(i).joinable()) {
      threads.at(i).join();
    }
  }
  return 0;
}