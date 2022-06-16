//
// Created by fss on 22-6-9.
//
#include "glog/logging.h"
#include "chain.h"
#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>

#include <vector>
#include <string>

int main(int argc, char *argv[]) {

  boost::program_options::options_description desc("Options");
  desc.add_options()
      ("help,h", "produce help message")
      ("rtsp", boost::program_options::value<std::string>(), "rtsp address");

  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
  }
  catch (boost::exception &e) {
    std::cerr << boost::diagnostic_information(e) << std::endl;
    return -1;
  }

  google::InitGoogleLogging(argv[0]);
//  FLAGS_log_dir = "./log";
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 1;
  std::vector<std::string> subscriptions;
  std::string rtsp =  vm["rtsp"].as<std::string>();
  std::vector<std::thread> threads;
  for (int i = 0; i < 1; ++i) {
    std::thread t([&, i] {
      VideoStream stream(i + 3,
                         3,
                         rtsp,
                         subscriptions);
      bool b = stream.Open();
//      stream.set_inference(8,"/home/fss/code/origin_vsc/tinyvcs/tmp/v5m8.plan");
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