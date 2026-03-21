// census_transform.h
// Census 变换模块，将灰度图像转换为 Census 特征图，支持 CPU 和 GPU (Vulkan) 后端
// 最后更新: 2026-03-21
// 作者: DeepSeek

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <atomic>
#include "census/filters.h"

namespace stereo_depth::census {

enum class Mode {
    CPU,
    GPU
};

enum class CensusMode {
    STANDARD,       // 传统 Census
    ADAPTIVE,       // 自适应阈值
    MEDIAN_CENTER   // 中值-Census（已并入 GPU 统一实现，保留枚举用于配置）
};

struct CensusConfig {
    // 窗口尺寸
    int window_width = 5;
    int window_height = 3;

    // GPU 工作组大小
    int workgroup_size_x = 32;
    int workgroup_size_y = 32;

    // ========== CPU 前滤波配置 ==========
    std::string cpu_pre_filter_type = "bilateral";   // "none", "median", "bilateral"
    int cpu_pre_median_ksize = 5;                    // 中值滤波核大小（奇数）
    int cpu_pre_bilateral_d = 9;                     // 双边滤波直径（奇数）
    double cpu_pre_bilateral_sigma_color = 50.0;
    double cpu_pre_bilateral_sigma_space = 9.0;

    // ========== 后滤波（CPU）配置 ==========
    std::string post_filter_type = "none";           // "none", "median", "bilateral"
    int post_filter_median_ksize = 3;
    int post_filter_bilateral_d = 5;
    double post_filter_bilateral_sigma_color = 50.0;
    double post_filter_bilateral_sigma_space = 5.0;

    // ========== GPU 前滤波配置（仅 GPU 后端有效）==========
    std::string gpu_filter_type = "bilateral";       // "none", "median", "bilateral"
    int gpu_bilateral_d = 15;                         // 双边滤波直径（奇数，支持最大 15，即窗口 15x15）
    double gpu_bilateral_sigma_color = 50.0;
    double gpu_bilateral_sigma_space = 9.0;

    // ========== Census 算法配置 ==========
    CensusMode census_mode = CensusMode::ADAPTIVE;
    int adaptive_threshold = 2;

    bool loadFromConfig();
};

class CensusTransform {
public:
    CensusTransform();
    explicit CensusTransform(const std::string& config_path);
    ~CensusTransform();

    CensusTransform(const CensusTransform&) = delete;
    CensusTransform& operator=(const CensusTransform&) = delete;
    CensusTransform(CensusTransform&&) = delete;
    CensusTransform& operator=(CensusTransform&&) = delete;

    bool initialize(int window_size = 5);
    bool initialize(int win_width, int win_height);
    bool initializeFromConfig();

    bool compute(const cv::Mat& src, cv::Mat& dst);

    Mode getMode() const { return m_mode; }
    void setMode(Mode mode);
    bool isInitialized() const { return m_initialized; }
    int getWindowWidth() const { return m_win_width; }
    int getWindowHeight() const { return m_win_height; }
    static bool isGpuAvailable();

private:
    bool computeCPU(const cv::Mat& src, cv::Mat& dst);
#ifdef WITH_VULKAN
    bool computeGPU(const cv::Mat& src, cv::Mat& dst);
    bool initGPU();
#endif

    Mode m_mode;
    int m_win_width;
    int m_win_height;
    int m_workgroup_x;
    int m_workgroup_y;
    std::atomic<bool> m_initialized;

    // CPU 前滤波参数
    FilterParams m_cpu_pre_filter;

    // 后滤波参数
    FilterParams m_post_filter;

    // GPU 前滤波参数
    std::string m_gpu_filter_type;
    int m_gpu_bilateral_d;
    double m_gpu_bilateral_sigma_color;
    double m_gpu_bilateral_sigma_space;

    // Census 算法参数
    CensusMode m_census_mode;
    int m_adaptive_threshold;

#ifdef WITH_VULKAN
    struct GpuResources;
    GpuResources* m_gpu = nullptr;
#endif

    CensusConfig m_config;
};

} // namespace stereo_depth::census
