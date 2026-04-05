#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::Preprocess() : m_censusGpu(nullptr) {
    LOG_INFO("Preprocess 构造函数");
}

// 析构函数在 preprocess_gpu.cpp 中定义（因为需要完整类型）

bool Preprocess::initFromConfig() {
    LOG_INFO("Preprocess::initFromConfig 开始");
    if (!m_denoiser.initFromConfig()) {
        LOG_ERROR("去噪器初始化失败");
        return false;
    }

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    CensusParams censusParams;
    censusParams.window_width = cfg.get<int>("preprocess.census.window_width", 5);
    censusParams.window_height = cfg.get<int>("preprocess.census.window_height", 3);
    censusParams.adaptive_threshold = cfg.get<int>("preprocess.census.adaptive_threshold", 2);
    censusParams.workgroup_size_x = cfg.get<int>("preprocess.gpu.workgroup_size_x", 16);
    censusParams.workgroup_size_y = cfg.get<int>("preprocess.gpu.workgroup_size_y", 16);
    if (!m_censusCpu.init(censusParams)) {
        LOG_ERROR("Census CPU 初始化失败");
        return false;
    }

    m_useGpuConfig = cfg.get<bool>("preprocess.preprocess_use_gpu", false);
    if (m_useGpuConfig) {
        LOG_INFO("配置要求使用 GPU 后端，尝试初始化 GPU Census 模块...");
        if (initGpu()) {
            m_gpuAvailable = true;
            LOG_INFO("GPU Census 模块初始化成功，将使用 GPU 加速 Census");
        } else {
            LOG_WARN("GPU Census 模块初始化失败，将回退到 CPU 后端");
            m_gpuAvailable = false;
        }
    } else {
        LOG_INFO("配置未要求使用 GPU，将使用 CPU 后端");
        m_gpuAvailable = false;
    }

    m_initialized = true;
    LOG_INFO("Preprocess 模块初始化完成");
    return true;
}

bool Preprocess::process(const cv::Mat& left, const cv::Mat& right,
                         cv::Mat& leftCensus, cv::Mat& rightCensus) {
    if (!m_initialized) {
        LOG_ERROR("Preprocess 未初始化");
        return false;
    }
    if (left.empty() || right.empty()) {
        LOG_ERROR("输入图像为空");
        return false;
    }
    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        LOG_ERROR("输入图像必须是 CV_8UC3 彩色图");
        return false;
    }

    if (m_gpuAvailable) {
        return processGpu(left, right, leftCensus, rightCensus);
    } else {
        return processCpu(left, right, leftCensus, rightCensus);
    }
}

bool Preprocess::denoiseOnly(const cv::Mat& src, cv::Mat& dst) {
    return m_denoiser.process(src, dst);
}

bool Preprocess::processCpu(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& leftCensus, cv::Mat& rightCensus) {
    LOG_DEBUG("CPU 处理开始");
    cv::Mat leftGray, rightGray;
    cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);

    cv::Mat leftDenoised, rightDenoised;
    if (!m_denoiser.process(leftGray, leftDenoised) ||
        !m_denoiser.process(rightGray, rightDenoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    if (!m_censusCpu.compute(leftDenoised, rightDenoised, leftCensus, rightCensus)) {
        LOG_ERROR("Census 计算失败");
        return false;
    }
    LOG_DEBUG("CPU 处理完成");
    return true;
}

// GPU 相关函数的实现都在 preprocess_gpu.cpp 中，这里不需要提供空实现

} // namespace stereo_depth::preprocess
