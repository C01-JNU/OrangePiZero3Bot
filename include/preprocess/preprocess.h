// preprocess.h
// 前处理主类：去噪 + Census 变换
// 最后更新: 2026-03-28

#pragma once

#include <opencv2/core.hpp>
#include "denoiser.h"
#include "census.h"

namespace stereo_depth::preprocess {

struct PreprocessConfig {
    DenoiseParams denoise;
    CensusParams census;
};

class Preprocess {
public:
    Preprocess();
    ~Preprocess();

    // 从配置文件初始化
    bool initFromConfig();

    // 处理双目图像对
    bool process(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& left_census, cv::Mat& right_census);

    // 单独获取去噪后的图像（可选）
    bool denoiseOnly(const cv::Mat& src, cv::Mat& dst);

private:
    Denoiser m_denoiser;
    CensusTransform m_census;
    bool m_initialized = false;
};

} // namespace stereo_depth::preprocess
