#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <string>

namespace stereo_depth {
namespace utils {

class Logger {
public:
    static void initialize(const std::string& name = "stereo_depth", 
                          spdlog::level::level_enum level = spdlog::level::info);
    
    static std::shared_ptr<spdlog::logger> getLogger();
    
    // 设置日志级别
    static void setLevel(spdlog::level::level_enum level);
    
    // 获取当前日志级别
    static spdlog::level::level_enum getLevel();
    
    // 检查是否启用某个级别
    static bool shouldLog(spdlog::level::level_enum level);
    
private:
    static std::shared_ptr<spdlog::logger> logger_;
    
    Logger() = delete;
    ~Logger() = delete;
};

// 方便使用的宏
#define LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)
#define LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)
#define LOG_INFO(...)     SPDLOG_LOGGER_INFO(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)
#define LOG_WARN(...)     SPDLOG_LOGGER_WARN(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)
#define LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(stereo_depth::utils::Logger::getLogger(), __VA_ARGS__)

} // namespace utils
} // namespace stereo_depth
