//
// Created by fss on 22-6-9.
//

#include <vector>
#include <string>

#include "boost/filesystem.hpp"
#include "fmt/core.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

#include "video_stream.h"

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
DEFINE_string(sub, "tcp://127.0.0.1:10080", "zeromq address");
DEFINE_bool(stderr, true, "log to stderr");
DEFINE_int32(loglevel, 1, "min log level");
DEFINE_int32(id, 0, "stream id");
DEFINE_int32(batch_size, 8, "inference batch size");
DEFINE_int32(duration, 3, "inference duration frame");
DEFINE_int32(width, 640, "inference image width");
DEFINE_int32(height, 640, "inference image height");

int main(int argc, char *argv[]) {
  namespace fs = boost::filesystem;

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(FLAGS_log.data());
  FLAGS_loglevel = 1;
  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;

  if (!fs::exists(FLAGS_log)) {
    bool has_create_dir = fs::create_directories(FLAGS_log);
    if (has_create_dir)
      LOG(INFO) << "create log dir: " << FLAGS_log;
  }

  VideoStream stream(FLAGS_id,
                     FLAGS_duration,
                     FLAGS_height,
                     FLAGS_width,
                     FLAGS_rtsp,
                     FLAGS_sub);
  if (!FLAGS_engine.empty()) {
    stream.set_inference(FLAGS_batch_size, FLAGS_engine);
  }
  bool b = stream.Open();
  if (!b) {
    LOG(FATAL) << "stream can be opened!";
    return -1;
  }
  stream.Run();
  return 0;
}