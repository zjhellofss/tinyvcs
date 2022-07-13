//
// Created by fss on 22-6-7.
//
#ifndef TINYVCS_WEBSOCKET_CLIENT_H_
#define TINYVCS_WEBSOCKET_CLIENT_H_
#include "glog/logging.h"
#include "zmq.hpp"

#include "frame.h"

class ClientChannel {
 public:
  explicit ClientChannel(const std::string &address) {
    this->address_ = address;
  }

  bool Init();

  bool PublishFrame(const Frame &frame);

 private:
  static std::string Pack(const Frame &frame);
 private:
  std::string address_;
  zmq::context_t ctx_;
  zmq::socket_t sock_;
  bool has_init_ = false;
};

#endif //TINYVCS_WEBSOCKET_CLIENT_H_
