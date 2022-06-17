//
// Created by fss on 22-6-7.
//
#ifndef TINYVCS_WEBSOCKET_CLIENT_H_
#define TINYVCS_WEBSOCKET_CLIENT_H_
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/common/memory.hpp>
typedef websocketpp::client<websocketpp::config::asio_client> client;
class ConnectionMeta {
 public:
  typedef websocketpp::lib::shared_ptr<ConnectionMeta> ptr;

  ConnectionMeta(websocketpp::connection_hdl hdl, std::string uri) : hdl_(std::move(hdl)), uri_(std::move(uri)) {}

  void OnOpen(client *c, websocketpp::connection_hdl hdl);

  void OnClose(client *c, websocketpp::connection_hdl hdl);

  void OnFail(client *c, websocketpp::connection_hdl hdl);

  void OnMessage(const websocketpp::connection_hdl &, const client::message_ptr &msg);

  bool IsRunnable() const {
    return this->is_runnable_;
  }

  const websocketpp::connection_hdl &GetHdl() const {
    return this->hdl_;
  }

 private:
  websocketpp::connection_hdl hdl_;
  std::string uri_;
  std::string error_reason_;
  std::string server_;
  std::atomic_bool is_runnable_ = false;

};

class Connection {
 public:
  Connection() {
    endpoint_.clear_access_channels(websocketpp::log::alevel::all);
    endpoint_.clear_error_channels(websocketpp::log::elevel::all);
    endpoint_.init_asio();
    endpoint_.start_perpetual();
    thread_ = websocketpp::lib::make_shared<websocketpp::lib::thread>(&client::run, &endpoint_);
  }

  ~Connection();

  bool Connect(const std::string &uri);

  bool Send(const std::string &message);

  bool is_runnable() const;

 private:
  client endpoint_;
  websocketpp::lib::shared_ptr<websocketpp::lib::thread> thread_;
  ConnectionMeta::ptr connection_meta_;
};

#endif //TINYVCS_WEBSOCKET_CLIENT_H_
