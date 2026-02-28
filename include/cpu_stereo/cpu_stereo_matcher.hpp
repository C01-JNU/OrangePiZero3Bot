#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace stereo_depth {
namespace cpu_stereo {

/**
 * @brief CPU立体匹配器，支持SGBM和BM算法
 * 
 * 从配置文件读取算法类型和参数，创建对应的OpenCV立体匹配器。
 */
class CpuStereoMatcher {
public:
    CpuStereoMatcher();
    ~CpuStereoMatcher();

    /**
     * @brief 从全局配置初始化匹配器
     * @param config_path 配置文件路径（可选，默认使用ConfigManager）
     * @return 是否成功
     */
    bool initializeFromConfig(const std::string& config_path = "");

    /**
     * @brief 计算视差图
     * @param left  左灰度图 (CV_8UC1)
     * @param right 右灰度图 (CV_8UC1)
     * @return 16位视差图 (CV_16S)，视差值为真实视差的16倍
     */
    cv::Mat compute(const cv::Mat& left, const cv::Mat& right);

    /**
     * @brief 获取最近一次计算的耗时（毫秒）
     */
    double getLastTimeMs() const { return last_time_ms_; }

private:
    cv::Ptr<cv::StereoMatcher> matcher_;
    std::string algorithm_;
    double last_time_ms_ = 0.0;

    // 通用参数
    int disparity_range_;
    int min_disparity_;
    int median_filter_size_;

    // 后处理参数（用于SGBM和BM）
    int uniqueness_ratio_ = 15;
    int speckle_window_size_ = 0;
    int speckle_range_ = 0;
    int disp12_max_diff_ = -1;
    int pre_filter_cap_ = 0;

    // SGBM特有参数
    struct SgbmParams {
        int block_size;
        double p1;
        double p2;
        bool mode;
    } sgbm_params_;

    // BM特有参数
    struct BmParams {
        int block_size;
        int pre_filter_type;
        int pre_filter_size;
        int pre_filter_cap;
        int texture_threshold;
        int uniqueness_ratio;
        int speckle_window_size;
        int speckle_range;
        bool try_small_disp;
    } bm_params_;

    bool createMatcher();
};

} // namespace cpu_stereo
} // namespace stereo_depth
