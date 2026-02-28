#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <type_traits>
#include <sstream>
#include <algorithm>
#include <limits>

namespace stereo_depth {
namespace utils {

class Config {
public:
    using Value = std::variant<int, unsigned int, float, double, bool, std::string>;
    
    Config() = default;
    ~Config() = default;
    
    bool loadFromFile(const std::string& filepath);
    bool loadFromString(const std::string& yaml_str);
    bool saveToFile(const std::string& filepath);
    
    // 获取配置值（模板定义在类内）
    template<typename T>
    T get(const std::string& key, const T& default_value = T()) const {
        auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            return default_value;
        }
        try {
            return convertValue<T>(it->second);
        } catch (const std::exception& e) {
            return default_value;
        }
    }
    
    template<typename T>
    void set(const std::string& key, const T& value) {
        config_map_[key] = Value(value);
    }
    
    bool has(const std::string& key) const;
    std::vector<std::string> getKeys() const;
    void merge(const Config& other);
    
private:
    std::unordered_map<std::string, Value> config_map_;
    YAML::Node yaml_root_;
    
    void parseYamlNode(const YAML::Node& node, const std::string& prefix = "");
    void setFromYaml(const std::string& key, const YAML::Node& node);
    
    // 类型转换辅助函数
    template<typename T>
    static T convertValue(const Value& var) {
        struct ValueVisitor {
            T operator()(int val) const {
                if constexpr (std::is_same_v<T, std::string>) return std::to_string(val);
                else if constexpr (std::is_same_v<T, unsigned int>) {
                    if (val >= 0) return static_cast<unsigned int>(val);
                    else throw std::bad_variant_access();
                } else return static_cast<T>(val);
            }
            T operator()(unsigned int val) const {
                if constexpr (std::is_same_v<T, std::string>) return std::to_string(val);
                else if constexpr (std::is_same_v<T, int>) {
                    if (val <= static_cast<unsigned int>(std::numeric_limits<int>::max()))
                        return static_cast<int>(val);
                    else throw std::bad_variant_access();
                } else return static_cast<T>(val);
            }
            T operator()(float val) const {
                if constexpr (std::is_same_v<T, std::string>) return std::to_string(val);
                else return static_cast<T>(val);
            }
            T operator()(double val) const {
                if constexpr (std::is_same_v<T, std::string>) return std::to_string(val);
                else return static_cast<T>(val);
            }
            T operator()(bool val) const {
                if constexpr (std::is_same_v<T, std::string>) return val ? "true" : "false";
                else return static_cast<T>(val);
            }
            T operator()(const std::string& val) const {
                if constexpr (std::is_same_v<T, std::string>) return val;
                else if constexpr (std::is_same_v<T, bool>) {
                    std::string lower_val = val;
                    std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);
                    if (lower_val == "true" || lower_val == "yes" || lower_val == "1") return true;
                    else if (lower_val == "false" || lower_val == "no" || lower_val == "0") return false;
                    else throw std::bad_variant_access();
                } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
                    std::stringstream ss(val);
                    T result;
                    ss >> result;
                    if (ss.fail()) throw std::bad_variant_access();
                    return result;
                } else throw std::bad_variant_access();
            }
        };
        return std::visit(ValueVisitor{}, var);
    }
};

class ConfigManager {
public:
    static ConfigManager& getInstance();
    
    bool loadGlobalConfig(const std::string& path = "config");
    Config& getConfig() { return config_; }
    const Config& getConfig() const { return config_; }
    bool reload();
    std::string getConfigPath() const { return config_path_; }
    
private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    
    Config config_;
    std::string config_path_;
};

} // namespace utils
} // namespace stereo_depth
