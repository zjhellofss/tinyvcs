//
// Created by fss on 22-6-9.
//
#include "json/jsonbuilder.h"
#include <json.hpp>
#include <glog/logging.h>
std::string create_json(const std::map<std::string, std::variant<float, int, std::string>> &kvs) {
  using json = nlohmann::json;
  json j;
  for (const auto &iter : kvs) {
    std::visit([&](const auto &val) {
      using T = std::decay_t<decltype(val)>;
      if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int> || std::is_same_v<T, std::string>) {
        j[iter.first] = val;
      }
      else{
        LOG(ERROR)<<"Error format in json";
      }
    }, iter.second);
  }
  std::string json_str = to_string(j);
  return json_str;
}
