//
// Created by fss on 22-6-9.
//

#ifndef TINYVCS_SOURCE_JSON_JSONBUILDER_H_
#define TINYVCS_SOURCE_JSON_JSONBUILDER_H_
#include <map>
#include <variant>
#include <string>
std::string create_json(const std::map<std::string, std::variant<float, int, std::string>> &kvs);
#endif //TINYVCS_SOURCE_JSON_JSONBUILDER_H_
