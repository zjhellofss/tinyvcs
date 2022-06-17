//
// Created by fss on 22-6-9.
//

#include <vector>
#include <string>
#include <fmt/core.h>

#include "glog/logging.h"
#include "chain.h"
#include "gflags/gflags.h"

static bool validateRtsp(const char *flag_name, const std::string &address) {
  if (address.empty()) {
    return false;
  }
  auto pos = address.find("rtsp://");
  if (pos != 0) {
    LOG(WARNING) << "invalid rtsp address: " << address;
    return false;
  } else {
    return true;
  }
}
DEFINE_string(rtsp, "", "Which rtsp address to connect");
DEFINE_validator(rtsp, &validateRtsp);

DEFINE_string(engine, "", "model engine for inference");
DEFINE_string(log, "./log", "log file dir");
DEFINE_bool(stderr, true, "log to stderr");
DEFINE_int32(loglevel, 0, "min log level");
DEFINE_int32(id, 0, "stream id");
DEFINE_int32(batch_size, 8, "inference batch size");
DEFINE_int32(duration, 3, "inference duration frame");

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(FLAGS_log.data());
  std::vector<std::string> subscriptions;
  VideoStream stream(FLAGS_id,
                     FLAGS_duration,
                     FLAGS_rtsp,
                     subscriptions);
  bool b = stream.Open();
  if (!FLAGS_engine.empty()) {
    stream.set_inference(FLAGS_batch_size, FLAGS_engine);
  }
  if (!b) {
    LOG(FATAL) << "stream can be opened!";
    return -1;
  }
  stream.Run();
  return 0;
}