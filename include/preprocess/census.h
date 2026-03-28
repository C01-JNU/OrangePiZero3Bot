// census.h
// 自适应阈值 Census 变换（CPU 实现）
// 从原 census 模块移植，去除滤波，仅保留变换逻辑
// 最后更新: 2026-03-28

#pragma once

#include <opencv2/core.hpp>

namespace stereo_depth::preprocess {

struct CensusParams {
    int window_width = 5;
    int window_height = 3;
    int adaptive_threshold = 2;   // 0 表示标准模式
};

class CensusTransform {
public:
    CensusTransform();
    ~CensusTransform();

    bool init(const CensusParams& params);
    bool compute(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& left_census, cv::Mat& right_census);

private:
    void computeOne(const cv::Mat& src, cv::Mat& dst);
    CensusParams m_params;
};

} // namespace stereo_depth::preprocess
