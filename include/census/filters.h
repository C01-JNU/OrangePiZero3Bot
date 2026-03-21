// filters.h
// 滤波算法模块，提供中值、双边滤波的 CPU 实现（预留 GPU 接口）
// 最后更新: 2026-03-21
// 作者: DeepSeek

#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace stereo_depth::census {

/**
 * @brief 滤波类型枚举
 */
enum class FilterType {
    NONE,       ///< 无滤波
    MEDIAN,     ///< 中值滤波
    BILATERAL   ///< 双边滤波
};

/**
 * @brief 滤波参数结构体
 */
struct FilterParams {
    FilterType type = FilterType::NONE;

    // 中值滤波参数
    int median_ksize = 3;   // 核大小（奇数）

    // 双边滤波参数
    int bilateral_d = 5;
    double bilateral_sigma_color = 50.0;
    double bilateral_sigma_space = 50.0;
};

/**
 * @brief 对图像应用滤波（CPU 实现）
 * @param src 输入图像 (CV_8UC1)
 * @param dst 输出图像，若为空则自动创建
 * @param params 滤波参数
 * @return 是否成功
 */
bool applyFilterCPU(const cv::Mat& src, cv::Mat& dst, const FilterParams& params);

/**
 * @brief 预留 GPU 接口（暂未实现）
 */
bool applyFilterGPU(const cv::Mat& src, cv::Mat& dst, const FilterParams& params);

} // namespace stereo_depth::census
