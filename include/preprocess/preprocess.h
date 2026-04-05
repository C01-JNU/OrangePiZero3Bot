// preprocess.h
// 前处理主类：去噪（CPU）+ Census 变换（可选 GPU），运行时选择后端
// 最后更新: 2026-04-05

#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include "denoiser.h"
#include "census.h"

namespace stereo_depth::preprocess {

class GpuCensusTransform;

/**
 * @brief 前处理主类，整合去噪和 Census 变换
 */
class Preprocess {
public:
    Preprocess();
    ~Preprocess();

    bool initFromConfig();
    bool process(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& leftCensus, cv::Mat& rightCensus);
    bool denoiseOnly(const cv::Mat& src, cv::Mat& dst);

    void* getFilteredImageHandle() const;
    int getFilteredImageWidth() const;
    int getFilteredImageHeight() const;

    bool isGpuAvailable() const { return m_gpuAvailable; }

private:
    bool processCpu(const cv::Mat& left, const cv::Mat& right,
                    cv::Mat& leftCensus, cv::Mat& rightCensus);
    bool processGpu(const cv::Mat& left, const cv::Mat& right,
                    cv::Mat& leftCensus, cv::Mat& rightCensus);
    bool initGpu();

    Denoiser m_denoiser;
    CensusTransform m_censusCpu;
    bool m_initialized = false;
    bool m_gpuAvailable = false;
    bool m_useGpuConfig = false;

    GpuCensusTransform* m_censusGpu;   // 原始指针，手动管理
};

} // namespace stereo_depth::preprocess
