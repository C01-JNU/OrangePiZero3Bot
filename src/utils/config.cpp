#include "utils/config.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <dirent.h>

namespace stereo_depth {
namespace utils {

bool Config::loadFromFile(const std::string& filepath) {
    try {
        if (!std::filesystem::exists(filepath)) {
            LOG_ERROR("Config file does not exist: {}", filepath);
            return false;
        }
        yaml_root_ = YAML::LoadFile(filepath);
        config_map_.clear();
        parseYamlNode(yaml_root_);
        LOG_INFO("Loaded config from: {}", filepath);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to load config file {}: {}", filepath, e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading config file {}: {}", filepath, e.what());
        return false;
    }
}

bool Config::loadFromString(const std::string& yaml_str) {
    try {
        yaml_root_ = YAML::Load(yaml_str);
        config_map_.clear();
        parseYamlNode(yaml_root_);
        LOG_INFO("Loaded config from string");
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to load config from string: {}", e.what());
        return false;
    }
}

bool Config::saveToFile(const std::string& filepath) {
    try {
        std::filesystem::path path(filepath);
        std::filesystem::create_directories(path.parent_path());
        std::ofstream fout(filepath);
        if (!fout.is_open()) {
            LOG_ERROR("Failed to open file for writing: {}", filepath);
            return false;
        }
        fout << yaml_root_;
        fout.close();
        LOG_INFO("Saved config to: {}", filepath);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save config to file {}: {}", filepath, e.what());
        return false;
    }
}

bool Config::has(const std::string& key) const {
    return config_map_.find(key) != config_map_.end();
}

std::vector<std::string> Config::getKeys() const {
    std::vector<std::string> keys;
    keys.reserve(config_map_.size());
    for (const auto& pair : config_map_) {
        keys.push_back(pair.first);
    }
    return keys;
}

void Config::merge(const Config& other) {
    for (const auto& pair : other.config_map_) {
        config_map_[pair.first] = pair.second;
    }
}

void Config::parseYamlNode(const YAML::Node& node, const std::string& prefix) {
    if (node.IsMap()) {
        for (const auto& pair : node) {
            std::string key = pair.first.as<std::string>();
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            if (pair.second.IsScalar()) {
                setFromYaml(full_key, pair.second);
            } else if (pair.second.IsSequence() || pair.second.IsMap()) {
                parseYamlNode(pair.second, full_key);
            }
        }
    } else if (node.IsSequence()) {
        for (size_t i = 0; i < node.size(); ++i) {
            std::string full_key = prefix + "[" + std::to_string(i) + "]";
            if (node[i].IsScalar()) {
                setFromYaml(full_key, node[i]);
            } else {
                parseYamlNode(node[i], full_key);
            }
        }
    } else if (node.IsScalar()) {
        setFromYaml(prefix, node);
    }
}

void Config::setFromYaml(const std::string& key, const YAML::Node& node) {
    try {
        if (node.IsNull()) return;
        try {
            int int_val = node.as<int>();
            config_map_[key] = Value(int_val);
            if (int_val >= 0) {
                std::string uint_key = key + ".uint";
                unsigned int uint_val = static_cast<unsigned int>(int_val);
                config_map_[uint_key] = Value(uint_val);
            }
            return;
        } catch (...) {}
        try {
            float float_val = node.as<float>();
            config_map_[key] = Value(float_val);
            return;
        } catch (...) {}
        try {
            double double_val = node.as<double>();
            config_map_[key] = Value(double_val);
            return;
        } catch (...) {}
        try {
            bool bool_val = node.as<bool>();
            config_map_[key] = Value(bool_val);
            return;
        } catch (...) {}
        std::string str_val = node.as<std::string>();
        config_map_[key] = Value(str_val);
    } catch (const std::exception& e) {
        LOG_WARN("Failed to parse config key {}: {}", key, e.what());
    }
}

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::loadGlobalConfig(const std::string& path) {
    config_path_ = path;
    
    std::filesystem::path fs_path(path);
    if (std::filesystem::is_directory(fs_path)) {
        std::vector<std::string> yaml_files;
        for (const auto& entry : std::filesystem::directory_iterator(fs_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".yaml" || ext == ".yml") {
                    yaml_files.push_back(entry.path().string());
                }
            }
        }
        std::sort(yaml_files.begin(), yaml_files.end());
        
        if (yaml_files.empty()) {
            LOG_ERROR("No YAML files found in directory: {}", path);
            return false;
        }
        
        Config temp_config;
        bool any_success = false;
        for (const auto& file : yaml_files) {
            if (temp_config.loadFromFile(file)) {
                config_.merge(temp_config);
                any_success = true;
            } else {
                LOG_WARN("Failed to load config file: {}", file);
            }
        }
        
        if (!any_success) {
            LOG_ERROR("Failed to load any config file from directory: {}", path);
            return false;
        }
        
        LOG_INFO("Loaded {} config files from directory: {}", yaml_files.size(), path);
        return true;
    } else {
        bool success = config_.loadFromFile(path);
        if (success) {
            LOG_INFO("Global config loaded successfully from: {}", path);
        } else {
            LOG_ERROR("Failed to load global config from: {}", path);
        }
        return success;
    }
}

bool ConfigManager::reload() {
    if (config_path_.empty()) {
        LOG_ERROR("No config path set for reload");
        return false;
    }
    LOG_INFO("Reloading config from: {}", config_path_);
    config_ = Config();
    return loadGlobalConfig(config_path_);
}

} // namespace utils
} // namespace stereo_depth
