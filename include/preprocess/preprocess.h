// preprocess.h
// 前处理主类：去噪 + Census 变换，支持 CPU 和 GPU (Vulkan) 后端
// 最后更新: 2026-03-29

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

    bool initFromConfig();
    bool process(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& left_census, cv::Mat& right_census);
    bool denoiseOnly(const cv::Mat& src, cv::Mat& dst);

private:
    bool processCPU(const cv::Mat& left, const cv::Mat& right,
                    cv::Mat& left_census, cv::Mat& right_census);
    bool processGPU(const cv::Mat& left, const cv::Mat& right,
                    cv::Mat& left_census, cv::Mat& right_census);
    bool initGPU();

    Denoiser m_denoiser;
    CensusTransform m_census;
    bool m_initialized = false;

#ifdef WITH_VULKAN
    struct GpuResources;
    GpuResources* m_gpu;
#endif
};

} // namespace stereo_depth::preprocess
