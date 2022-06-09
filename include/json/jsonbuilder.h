#include <string>
#include <map>
#include <variant>
typedef std::map<std::string, std::variant<float, int, double, std::string>> vmaps;
std::string create_json(const vmaps &kvs);