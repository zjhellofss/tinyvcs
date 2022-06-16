#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/memory.hpp>

#include <iostream>
#include <string>
#include <glog/logging.h>
#include <fmt/core.h>

#include "websocket/client.h"

void ConnectionMeta::OnOpen(client *c, websocketpp::connection_hdl hdl) {
  is_runnable_ = true; //连接打开
  client::connection_ptr con = c->get_con_from_hdl(hdl);
  server_ = con->get_response_header("Server");
  LOG(INFO) << "Connection open success from: " << server_;
}

void ConnectionMeta::OnClose(client *c, websocketpp::connection_hdl hdl) {
  is_runnable_ = false; //连接关闭
  client::connection_ptr con = c->get_con_from_hdl(hdl);
  error_reason_ = fmt::format("close code: {}, close reason: {} server: {}",
                              con->get_remote_close_code(),
                              con->get_remote_close_reason(), server_);
  LOG(INFO) << error_reason_;
}

void ConnectionMeta::OnFail(client *c, websocketpp::connection_hdl hdl) {
  is_runnable_ = false;
  client::connection_ptr con = c->get_con_from_hdl(hdl);
  server_ = con->get_response_header("Server");
  error_reason_ = con->get_ec().message();
  LOG(FATAL) << error_reason_;
}

void ConnectionMeta::OnMessage(const websocketpp::connection_hdl &,
                               const std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager>> &msg) {
}

Connection::~Connection() {
  endpoint_.stop_perpetual();
  if (connection_meta_ && connection_meta_->IsRunnable()) {
    LOG(INFO) << "Closing connection...";
    websocketpp::lib::error_code ec;
    endpoint_.close(connection_meta_->GetHdl(), websocketpp::close::status::going_away, "", ec);
    if (ec) {
      LOG(ERROR) << "Error closing connection: " << ec.message();
    }
  }
  if (thread_->joinable()) {
    thread_->join();
  }
}

bool Connection::Connect(const std::string &uri) {
  websocketpp::lib::error_code ec;
  client::connection_ptr con = endpoint_.get_connection(uri, ec);

  if (ec) {
    LOG(ERROR) << "Connect initialization error: " << ec.message();
    return false;
  }

  connection_meta_ = websocketpp::lib::make_shared<ConnectionMeta>(con->get_handle(), uri);

  con->set_open_handler(websocketpp::lib::bind(
      &ConnectionMeta::OnOpen,
      connection_meta_,
      &endpoint_,
      websocketpp::lib::placeholders::_1
  ));
  con->set_fail_handler(websocketpp::lib::bind(
      &ConnectionMeta::OnFail,
      connection_meta_,
      &endpoint_,
      websocketpp::lib::placeholders::_1
  ));
  con->set_close_handler(websocketpp::lib::bind(
      &ConnectionMeta::OnClose,
      connection_meta_,
      &endpoint_,
      websocketpp::lib::placeholders::_1
  ));
  con->set_message_handler(websocketpp::lib::bind(
      &ConnectionMeta::OnMessage,
      connection_meta_,
      websocketpp::lib::placeholders::_1,
      websocketpp::lib::placeholders::_2
  ));

  endpoint_.connect(con);
  return true;
}

bool Connection::Send(const std::string &message) {
  websocketpp::lib::error_code ec;
  if (!connection_meta_ || !connection_meta_->IsRunnable()) {
    LOG(ERROR) << "No connection found";
    return false;
  } else {
    endpoint_.send(connection_meta_->GetHdl(), message, websocketpp::frame::opcode::text, ec);
    if (ec) {
      LOG(ERROR) << "Error sending message: " << ec.message();
      return false;
    } else {
      LOG(INFO) << "Send message successfully";
      return true;
    }
  }
}
bool Connection::is_runnable() const {
  return connection_meta_->IsRunnable();
}
