#include <string>
#include <map>
#include <variant>
std::string create_json(const std::map<std::string, std::variant<float, int, std::string>> &kvs);