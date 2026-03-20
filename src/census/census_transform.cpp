#include "census/census_transform.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::census {

bool CensusConfig::loadFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    window_width = cfg.get<int>("census.window_width", 5);
    window_height = cfg.get<int>("census.window_height", 5);
    workgroup_size_x = cfg.get<int>("census.workgroup_size_x", 32);
    workgroup_size_y = cfg.get<int>("census.workgroup_size_y", 32);

    filter_type = cfg.get<std::string>("census.filter_type", "none");
    filter_kernel_size = cfg.get<int>("census.filter_kernel_size", 3);
    filter_sigma = cfg.get<double>("census.filter_sigma", 0.5);

    if (window_width % 2 == 0) ++window_width;
    if (window_height % 2 == 0) ++window_height;
    if (filter_kernel_size % 2 == 0) ++filter_kernel_size;

    return true;
}

CensusTransform::CensusTransform()
    : m_win_width(5)
    , m_win_height(5)
    , m_workgroup_x(32)
    , m_workgroup_y(32)
    , m_initialized(false)
    , m_filter_type("none")
    , m_filter_kernel_size(3)
    , m_filter_sigma(0.5) {
#ifdef WITH_VULKAN
    m_mode = Mode::GPU;
    LOG_DEBUG("CensusTransform created, default mode: GPU (WITH_VULKAN)");
#else
    m_mode = Mode::CPU;
    LOG_DEBUG("CensusTransform created, default mode: CPU");
#endif
}

CensusTransform::CensusTransform(const std::string& config_path)
    : CensusTransform() {
    (void)config_path;
}

#ifndef WITH_VULKAN
CensusTransform::~CensusTransform() = default;
#endif

bool CensusTransform::initialize(int window_size) {
    return initialize(window_size, window_size);
}

bool CensusTransform::initialize(int win_width, int win_height) {
    if (win_width % 2 == 0 || win_height % 2 == 0) {
        LOG_ERROR("Window dimensions must be odd: {}x{}", win_width, win_height);
        return false;
    }
    m_win_width = win_width;
    m_win_height = win_height;
    m_initialized = true;
    LOG_INFO("CensusTransform initialized: window {}x{}, mode {}",
             m_win_width, m_win_height, m_mode == Mode::CPU ? "CPU" : "GPU");
    return true;
}

bool CensusTransform::initializeFromConfig() {
    if (!m_config.loadFromConfig()) {
        LOG_ERROR("Failed to load Census config");
        return false;
    }
    m_workgroup_x = m_config.workgroup_size_x;
    m_workgroup_y = m_config.workgroup_size_y;
    m_filter_type = m_config.filter_type;
    m_filter_kernel_size = m_config.filter_kernel_size;
    m_filter_sigma = m_config.filter_sigma;

    LOG_INFO("Census workgroup size: {}x{}, filter: {} (kernel={}, sigma={})",
             m_workgroup_x, m_workgroup_y, m_filter_type,
             m_filter_kernel_size, m_filter_sigma);
    return initialize(m_config.window_width, m_config.window_height);
}

void CensusTransform::setMode(Mode mode) {
    if (mode != m_mode) {
        LOG_INFO("Switching Census mode: {} -> {}",
                 m_mode == Mode::CPU ? "CPU" : "GPU",
                 mode == Mode::CPU ? "CPU" : "GPU");
        m_mode = mode;
    }
}

bool CensusTransform::isGpuAvailable() {
#ifdef WITH_VULKAN
    return true;
#else
    return false;
#endif
}

bool CensusTransform::compute(const cv::Mat& src, cv::Mat& dst) {
    if (!m_initialized) {
        LOG_ERROR("CensusTransform not initialized");
        return false;
    }
    if (src.empty()) {
        LOG_ERROR("Input image is empty");
        return false;
    }
    if (src.type() != CV_8UC1) {
        LOG_ERROR("Input image must be CV_8UC1, got {}", src.type());
        return false;
    }

    // 可选滤波
    cv::Mat processed = src;
    if (m_filter_type == "gaussian") {
        cv::GaussianBlur(src, processed,
                         cv::Size(m_filter_kernel_size, m_filter_kernel_size),
                         m_filter_sigma);
    } else if (m_filter_type == "median") {
        cv::medianBlur(src, processed, m_filter_kernel_size);
    } // else "none" 不做处理

    if (m_mode == Mode::GPU) {
#ifdef WITH_VULKAN
        if (!m_gpu) {
            if (!initGPU()) {
                LOG_ERROR("GPU initialization failed, falling back to CPU");
                m_mode = Mode::CPU;
                return computeCPU(processed, dst);
            }
        }
        return computeGPU(processed, dst);
#else
        LOG_WARN("GPU support not compiled, falling back to CPU");
        return computeCPU(processed, dst);
#endif
    } else {
        return computeCPU(processed, dst);
    }
}

} // namespace stereo_depth::census
