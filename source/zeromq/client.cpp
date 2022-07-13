#include "zeromq/client.h"

bool ClientChannel::Init() {
  if (this->address_.empty()) {
    LOG(ERROR) << "zeromq publish address is empty";
    return false;
  }
  sock_ = zmq::socket_t(ctx_, zmq::socket_type::pub);
  try {
    sock_.bind(this->address_);
  }
  catch (std::exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }

  LOG(ERROR) << "zeromq connect failed!";
  if (!sock_.handle())
    return false;

  has_init_ = true;
  return true;
}

bool ClientChannel::PublishFrame(const Frame &frame) {
  if (!has_init_) {
    if (!Init())
      return false;
  }
  try {
    std::string packed_data = Pack(frame);
    zmq::message_t payload(packed_data.begin(), packed_data.end());
    auto rc = sock_.send(payload, zmq::send_flags::dontwait);
    if (rc < 0) {
      return false;
    }
  }
  catch (std::exception &e) {
    LOG(ERROR) << e.what();
    return false;
  }
  return true;
}

std::string ClientChannel::Pack(const Frame &frame) {
  std::stringstream stream;
  msgpack::pack(stream, frame);
  std::string packed_data = stream.str();
  return packed_data;
}
