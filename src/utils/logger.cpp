#include "utils/logger.hpp"
#include <iostream>

namespace stereo_depth {
namespace utils {

std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

void Logger::initialize(const std::string& name, spdlog::level::level_enum level) {
    if (logger_ != nullptr) {
        // 已经初始化
        return;
    }
    
    try {
        // 创建控制台sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        
        // 设置日志格式
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
        
        // 创建logger
        logger_ = std::make_shared<spdlog::logger>(name, console_sink);
        
        // 设置日志级别
        logger_->set_level(level);
        
        // 设置刷新级别
        logger_->flush_on(spdlog::level::err);
        
        // 注册logger
        spdlog::register_logger(logger_);
        
        // 设置为默认logger
        spdlog::set_default_logger(logger_);
        
        LOG_INFO("Logger initialized with level: {}", spdlog::level::to_string_view(level));
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

std::shared_ptr<spdlog::logger> Logger::getLogger() {
    if (logger_ == nullptr) {
        // 如果没有初始化，使用默认配置初始化
        initialize();
    }
    return logger_;
}

void Logger::setLevel(spdlog::level::level_enum level) {
    getLogger()->set_level(level);
    LOG_INFO("Log level changed to: {}", spdlog::level::to_string_view(level));
}

spdlog::level::level_enum Logger::getLevel() {
    return getLogger()->level();
}

bool Logger::shouldLog(spdlog::level::level_enum level) {
    return getLogger()->should_log(level);
}

} // namespace utils
} // namespace stereo_depth
