#include "census/census_transform.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "census/filters.h"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::census {

bool CensusConfig::loadFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();

    window_width = cfg.get<int>("census.window_width", 5);
    window_height = cfg.get<int>("census.window_height", 3);
    workgroup_size_x = cfg.get<int>("census.workgroup_size_x", 32);
    workgroup_size_y = cfg.get<int>("census.workgroup_size_y", 32);

    // CPU 前滤波
    cpu_pre_filter_type = cfg.get<std::string>("census.cpu_pre_filter_type", "bilateral");
    cpu_pre_median_ksize = cfg.get<int>("census.cpu_pre_median_ksize", 5);
    cpu_pre_bilateral_d = cfg.get<int>("census.cpu_pre_bilateral_d", 9);
    cpu_pre_bilateral_sigma_color = cfg.get<double>("census.cpu_pre_bilateral_sigma_color", 50.0);
    cpu_pre_bilateral_sigma_space = cfg.get<double>("census.cpu_pre_bilateral_sigma_space", 9.0);

    // 后滤波
    post_filter_type = cfg.get<std::string>("census.post_filter_type", "none");
    post_filter_median_ksize = cfg.get<int>("census.post_filter_median_ksize", 3);
    post_filter_bilateral_d = cfg.get<int>("census.post_filter_bilateral_d", 5);
    post_filter_bilateral_sigma_color = cfg.get<double>("census.post_filter_bilateral_sigma_color", 50.0);
    post_filter_bilateral_sigma_space = cfg.get<double>("census.post_filter_bilateral_sigma_space", 5.0);

    // GPU 前滤波
    gpu_filter_type = cfg.get<std::string>("census.gpu_filter_type", "bilateral");
    gpu_bilateral_d = cfg.get<int>("census.gpu_bilateral_d", 15);
    gpu_bilateral_sigma_color = cfg.get<double>("census.gpu_bilateral_sigma_color", 50.0);
    gpu_bilateral_sigma_space = cfg.get<double>("census.gpu_bilateral_sigma_space", 9.0);

    // Census 模式
    std::string mode_str = cfg.get<std::string>("census.census_mode", "adaptive");
    if (mode_str == "standard") {
        census_mode = CensusMode::STANDARD;
    } else if (mode_str == "median_center") {
        census_mode = CensusMode::MEDIAN_CENTER;
    } else {
        census_mode = CensusMode::ADAPTIVE;
    }
    adaptive_threshold = cfg.get<int>("census.adaptive_threshold", 2);

    // 参数奇偶性修正
    if (window_width % 2 == 0) ++window_width;
    if (window_height % 2 == 0) ++window_height;
    if (cpu_pre_median_ksize % 2 == 0) ++cpu_pre_median_ksize;
    if (cpu_pre_bilateral_d > 0 && cpu_pre_bilateral_d % 2 == 0) ++cpu_pre_bilateral_d;
    if (post_filter_median_ksize % 2 == 0) ++post_filter_median_ksize;
    if (post_filter_bilateral_d > 0 && post_filter_bilateral_d % 2 == 0) ++post_filter_bilateral_d;
    if (gpu_bilateral_d > 0 && gpu_bilateral_d % 2 == 0) ++gpu_bilateral_d;

    return true;
}

CensusTransform::CensusTransform()
    : m_win_width(5)
    , m_win_height(3)
    , m_workgroup_x(32)
    , m_workgroup_y(32)
    , m_initialized(false)
    , m_gpu_filter_type("bilateral")
    , m_gpu_bilateral_d(15)
    , m_gpu_bilateral_sigma_color(50.0)
    , m_gpu_bilateral_sigma_space(9.0)
    , m_census_mode(CensusMode::ADAPTIVE)
    , m_adaptive_threshold(2) {
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

    // CPU 前滤波
    if (m_config.cpu_pre_filter_type == "median") {
        m_cpu_pre_filter.type = FilterType::MEDIAN;
        m_cpu_pre_filter.median_ksize = m_config.cpu_pre_median_ksize;
    } else if (m_config.cpu_pre_filter_type == "bilateral") {
        m_cpu_pre_filter.type = FilterType::BILATERAL;
        m_cpu_pre_filter.bilateral_d = m_config.cpu_pre_bilateral_d;
        m_cpu_pre_filter.bilateral_sigma_color = m_config.cpu_pre_bilateral_sigma_color;
        m_cpu_pre_filter.bilateral_sigma_space = m_config.cpu_pre_bilateral_sigma_space;
    } else {
        m_cpu_pre_filter.type = FilterType::NONE;
    }

    // 后滤波
    if (m_config.post_filter_type == "median") {
        m_post_filter.type = FilterType::MEDIAN;
        m_post_filter.median_ksize = m_config.post_filter_median_ksize;
    } else if (m_config.post_filter_type == "bilateral") {
        m_post_filter.type = FilterType::BILATERAL;
        m_post_filter.bilateral_d = m_config.post_filter_bilateral_d;
        m_post_filter.bilateral_sigma_color = m_config.post_filter_bilateral_sigma_color;
        m_post_filter.bilateral_sigma_space = m_config.post_filter_bilateral_sigma_space;
    } else {
        m_post_filter.type = FilterType::NONE;
    }

    // GPU 前滤波参数
    m_gpu_filter_type = m_config.gpu_filter_type;
    m_gpu_bilateral_d = m_config.gpu_bilateral_d;
    m_gpu_bilateral_sigma_color = m_config.gpu_bilateral_sigma_color;
    m_gpu_bilateral_sigma_space = m_config.gpu_bilateral_sigma_space;

    // Census 模式
    m_census_mode = m_config.census_mode;
    m_adaptive_threshold = m_config.adaptive_threshold;

    LOG_INFO("Census workgroup size: {}x{}, CPU pre_filter: type={}, GPU filter: {} (d={}, sc={}, ss={}), "
             "post_filter: type={}, census_mode={}, adaptive_threshold={}",
             m_workgroup_x, m_workgroup_y,
             static_cast<int>(m_cpu_pre_filter.type),
             m_gpu_filter_type, m_gpu_bilateral_d,
             m_gpu_bilateral_sigma_color, m_gpu_bilateral_sigma_space,
             static_cast<int>(m_post_filter.type),
             m_census_mode == CensusMode::STANDARD ? "standard" : (m_census_mode == CensusMode::ADAPTIVE ? "adaptive" : "median_center"),
             m_adaptive_threshold);

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

    // CPU 后端
    if (m_mode == Mode::CPU) {
        cv::Mat preprocessed;
        // 应用 CPU 前滤波
        if (!applyFilterCPU(src, preprocessed, m_cpu_pre_filter)) {
            LOG_ERROR("CPU pre-filter failed");
            return false;
        }
        // 计算 Census
        if (!computeCPU(preprocessed, dst)) return false;
    }
    // GPU 后端
    else {
#ifdef WITH_VULKAN
        if (!m_gpu) {
            if (!initGPU()) {
                LOG_ERROR("GPU initialization failed, falling back to CPU");
                m_mode = Mode::CPU;
                // 回退时也要应用 CPU 前滤波
                cv::Mat preprocessed;
                if (!applyFilterCPU(src, preprocessed, m_cpu_pre_filter)) {
                    LOG_ERROR("CPU pre-filter failed during fallback");
                    return false;
                }
                if (!computeCPU(preprocessed, dst)) return false;
            } else {
                if (!computeGPU(src, dst)) return false;
            }
        } else {
            if (!computeGPU(src, dst)) return false;
        }
#else
        LOG_ERROR("GPU mode selected but Vulkan not compiled");
        return false;
#endif
    }

    // 后滤波（对 Census 图）
    if (m_post_filter.type != FilterType::NONE) {
        cv::Mat filtered;
        // 将 CV_16U 转为 CV_8U（右移8位，保留高8位）
        cv::Mat dst_8u;
        dst.convertTo(dst_8u, CV_8U, 1.0 / 256.0);
        if (!applyFilterCPU(dst_8u, filtered, m_post_filter)) {
            LOG_ERROR("Post-filter failed");
            return false;
        }
        // 转回 CV_16U（左移8位）
        filtered.convertTo(dst, CV_16U, 256.0);
    }

    return true;
}

} // namespace stereo_depth::census
