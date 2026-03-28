// denoiser.h
// 去噪器接口，支持中值滤波、双边滤波（CPU 实现）
// 最后更新: 2026-03-28

#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace stereo_depth::preprocess {

enum class DenoiseMethod {
    NONE,
    MEDIAN,
    BILATERAL
};

struct DenoiseParams {
    DenoiseMethod method = DenoiseMethod::NONE;

    // 中值滤波参数
    int median_ksize = 3;

    // 双边滤波参数
    int bilateral_d = 9;
    double bilateral_sigma_color = 50.0;
    double bilateral_sigma_space = 9.0;
};

class Denoiser {
public:
    Denoiser();
    ~Denoiser();

    // 从配置文件初始化
    bool initFromConfig();

    // 设置参数
    bool setParams(const DenoiseParams& params);
    // 对单张灰度图去噪
    bool process(const cv::Mat& src, cv::Mat& dst);

private:
    DenoiseParams m_params;
};

} // namespace stereo_depth::preprocess
