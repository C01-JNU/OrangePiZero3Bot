// census.h
// 自适应阈值 Census 变换（CPU 实现）
// 最后更新: 2026-03-29

#pragma once

#include <opencv2/core.hpp>

namespace stereo_depth::preprocess {

struct CensusParams {
    int window_width = 5;
    int window_height = 3;
    int adaptive_threshold = 2;
    int workgroup_size_x = 16;   // 仅 GPU 使用
    int workgroup_size_y = 16;   // 仅 GPU 使用
};

class CensusTransform {
public:
    CensusTransform();
    ~CensusTransform();

    bool init(const CensusParams& params);
    bool compute(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& left_census, cv::Mat& right_census);

    // 获取参数（供 GPU 后端使用）
    int getWindowWidth() const { return m_params.window_width; }
    int getWindowHeight() const { return m_params.window_height; }
    int getAdaptiveThreshold() const { return m_params.adaptive_threshold; }
    int getWorkgroupX() const { return m_params.workgroup_size_x; }
    int getWorkgroupY() const { return m_params.workgroup_size_y; }

private:
    void computeOne(const cv::Mat& src, cv::Mat& dst);
    CensusParams m_params;
};

} // namespace stereo_depth::preprocess
