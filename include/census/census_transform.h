// census_transform.h
// Census 变换模块，将灰度图像转换为 Census 特征图，支持 CPU 和 GPU (Vulkan) 后端
// 最后更新: 2026-03-20
// 作者: DeepSeek

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <atomic>

namespace stereo_depth::census {

/**
 * @brief Census 变换运行模式
 */
enum class Mode {
    CPU,    ///< CPU 实现
    GPU     ///< GPU (Vulkan) 实现
};

/**
 * @brief Census 变换配置结构体
 */
struct CensusConfig {
    int window_width = 5;               ///< 窗口宽度（奇数）
    int window_height = 5;              ///< 窗口高度（奇数）
    int workgroup_size_x = 32;          ///< GPU 工作组宽度（需与着色器同步）
    int workgroup_size_y = 32;          ///< GPU 工作组高度（需与着色器同步）

    std::string filter_type = "none";   ///< 滤波类型: "none", "gaussian", "median"
    int filter_kernel_size = 3;         ///< 滤波核大小（奇数）
    double filter_sigma = 0.5;          ///< 标准差（仅用于高斯滤波）

    /**
     * @brief 从全局配置加载参数
     * @return 是否成功
     */
    bool loadFromConfig();
};

/**
 * @brief Census 变换类
 */
class CensusTransform {
public:
    CensusTransform();
    explicit CensusTransform(const std::string& config_path);
    ~CensusTransform();

    // 禁止拷贝，允许移动
    CensusTransform(const CensusTransform&) = delete;
    CensusTransform& operator=(const CensusTransform&) = delete;
    CensusTransform(CensusTransform&&) = delete;
    CensusTransform& operator=(CensusTransform&&) = delete;

    /**
     * @brief 初始化变换器（正方形窗口）
     * @param window_size 窗口尺寸（奇数）
     * @return 是否成功
     */
    bool initialize(int window_size = 5);

    /**
     * @brief 初始化变换器（矩形窗口）
     * @param win_width 窗口宽度（奇数）
     * @param win_height 窗口高度（奇数）
     * @return 是否成功
     */
    bool initialize(int win_width, int win_height);

    /**
     * @brief 从配置文件初始化
     * @return 是否成功
     */
    bool initializeFromConfig();

    /**
     * @brief 计算 Census 变换
     * @param src 输入灰度图像 (CV_8UC1)
     * @param dst 输出 Census 特征图 (CV_16U)
     * @return 是否成功
     */
    bool compute(const cv::Mat& src, cv::Mat& dst);

    /**
     * @brief 获取当前运行模式
     */
    Mode getMode() const { return m_mode; }

    /**
     * @brief 设置运行模式
     * @param mode 新模式
     */
    void setMode(Mode mode);

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief 获取窗口宽度
     */
    int getWindowWidth() const { return m_win_width; }

    /**
     * @brief 获取窗口高度
     */
    int getWindowHeight() const { return m_win_height; }

    /**
     * @brief 检查 GPU 是否可用（编译时定义 WITH_VULKAN）
     */
    static bool isGpuAvailable();

private:
    // CPU 实现
    bool computeCPU(const cv::Mat& src, cv::Mat& dst);

#ifdef WITH_VULKAN
    // GPU 实现
    bool computeGPU(const cv::Mat& src, cv::Mat& dst);
    bool initGPU();
#endif

    // 成员变量
    Mode m_mode;                     ///< 当前运行模式
    int m_win_width;                 ///< 窗口宽度
    int m_win_height;                ///< 窗口高度
    int m_workgroup_x;               ///< GPU 工作组宽度
    int m_workgroup_y;               ///< GPU 工作组高度
    std::atomic<bool> m_initialized; ///< 初始化标志

    // 滤波参数
    std::string m_filter_type;       ///< 滤波类型
    int m_filter_kernel_size;        ///< 滤波核大小
    double m_filter_sigma;           ///< 标准差（仅用于高斯滤波）

#ifdef WITH_VULKAN
    // GPU 资源（前向声明）
    struct GpuResources;
    GpuResources* m_gpu = nullptr;
#endif

    CensusConfig m_config;           ///< 运行时配置
};

} // namespace stereo_depth::census
