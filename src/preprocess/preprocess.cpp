#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::Preprocess()
#ifdef WITH_VULKAN
    : m_gpu(nullptr)
#endif
{
    LOG_INFO("Preprocess 构造函数, m_gpu={}", reinterpret_cast<void*>(m_gpu));
}

#ifndef WITH_VULKAN
Preprocess::~Preprocess() = default;
#endif

bool Preprocess::initFromConfig() {
    LOG_INFO("Preprocess::initFromConfig 开始");
    if (!m_denoiser.initFromConfig()) {
        LOG_ERROR("去噪器初始化失败");
        return false;
    }

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    CensusParams census_params;
    census_params.window_width = cfg.get<int>("preprocess.census.window_width", 5);
    census_params.window_height = cfg.get<int>("preprocess.census.window_height", 3);
    census_params.adaptive_threshold = cfg.get<int>("preprocess.census.adaptive_threshold", 2);
    census_params.workgroup_size_x = cfg.get<int>("preprocess.gpu.workgroup_size_x", 16);
    census_params.workgroup_size_y = cfg.get<int>("preprocess.gpu.workgroup_size_y", 16);
    if (!m_census.init(census_params)) {
        LOG_ERROR("Census初始化失败");
        return false;
    }

#ifdef WITH_VULKAN
    LOG_INFO("WITH_VULKAN已定义，将使用GPU后端");
#else
    LOG_INFO("WITH_VULKAN未定义，使用CPU后端");
#endif

    m_initialized = true;
    LOG_INFO("Preprocess模块初始化完成");
    return true;
}

bool Preprocess::process(const cv::Mat& left, const cv::Mat& right,
                         cv::Mat& left_census, cv::Mat& right_census) {
    if (!m_initialized) {
        LOG_ERROR("Preprocess未初始化");
        return false;
    }
    if (left.empty() || right.empty()) {
        LOG_ERROR("输入图像为空");
        return false;
    }
    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        LOG_ERROR("输入图像必须是CV_8UC3彩色图");
        return false;
    }

#ifdef WITH_VULKAN
    LOG_INFO("进入WITH_VULKAN分支, m_gpu={}", reinterpret_cast<void*>(m_gpu));
    if (!m_gpu) {
        LOG_INFO("m_gpu为空，初始化GPU...");
        if (!initGPU()) {
            LOG_ERROR("GPU初始化失败，回退到CPU");
            return processCPU(left, right, left_census, right_census);
        }
        LOG_INFO("GPU初始化成功, m_gpu={}", reinterpret_cast<void*>(m_gpu));
    }
    LOG_INFO("调用processGPU");
    return processGPU(left, right, left_census, right_census);
#else
    return processCPU(left, right, left_census, right_census);
#endif
}

bool Preprocess::denoiseOnly(const cv::Mat& src, cv::Mat& dst) {
    return m_denoiser.process(src, dst);
}

bool Preprocess::processCPU(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& left_census, cv::Mat& right_census) {
    LOG_INFO("CPU处理开始");
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    cv::Mat left_denoised, right_denoised;
    if (!m_denoiser.process(left_gray, left_denoised) ||
        !m_denoiser.process(right_gray, right_denoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    if (!m_census.compute(left_denoised, right_denoised, left_census, right_census)) {
        LOG_ERROR("Census计算失败");
        return false;
    }
    LOG_INFO("CPU处理完成");
    return true;
}

} // namespace stereo_depth::preprocess
