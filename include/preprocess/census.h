// Copyright (C) 2026 C01-JNU
// SPDX-License-Identifier: GPL-3.0-only
//
// This file is part of FishTotem.
//
// FishTotem is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FishTotem is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FishTotem. If not, see <https://www.gnu.org/licenses/>.


// census.h
// 自适应阈值 Census / Rank 变换（CPU 实现）
// 最后更新: 2026-04-06
// 作者: C01-JNU & AI助手

#pragma once

#include <opencv2/core.hpp>

namespace stereo_depth::preprocess {

/**
 * @brief 变换类型枚举
 */
enum class TransformType {
    CENSUS,   ///< Census 变换（输出比特编码）
    RANK      ///< Rank 变换（输出计数值）
};

/**
 * @brief Census / Rank 变换参数
 */
struct CensusParams {
    int window_width = 5;               ///< 窗口宽度（奇数）
    int window_height = 3;              ///< 窗口高度（奇数）
    int adaptive_threshold = 2;         ///< 自适应阈值（0 表示禁用）
    int workgroup_size_x = 16;          ///< GPU 工作组大小 X
    int workgroup_size_y = 16;          ///< GPU 工作组大小 Y
    TransformType transform_type = TransformType::CENSUS;  ///< 变换类型
};

/**
 * @brief CPU 实现的 Census / Rank 变换类
 */
class CensusTransform {
public:
    CensusTransform();
    ~CensusTransform();

    /**
     * @brief 初始化变换器
     * @param params 变换参数
     * @return 成功返回 true
     */
    bool init(const CensusParams& params);

    /**
     * @brief 对左右目灰度图执行变换
     * @param left 左目灰度图 (CV_8UC1)
     * @param right 右目灰度图 (CV_8UC1)
     * @param leftCensus 输出左目变换图 (CV_16U)
     * @param rightCensus 输出右目变换图 (CV_16U)
     * @return 成功返回 true
     */
    bool compute(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& leftCensus, cv::Mat& rightCensus);

    // 参数获取接口（供 GPU 后端使用）
    int getWindowWidth() const { return m_params.window_width; }
    int getWindowHeight() const { return m_params.window_height; }
    int getAdaptiveThreshold() const { return m_params.adaptive_threshold; }
    int getWorkgroupX() const { return m_params.workgroup_size_x; }
    int getWorkgroupY() const { return m_params.workgroup_size_y; }
    TransformType getTransformType() const { return m_params.transform_type; }

private:
    void computeOneCensus(const cv::Mat& src, cv::Mat& dst);
    void computeOneRank(const cv::Mat& src, cv::Mat& dst);
    CensusParams m_params;
};

} // namespace stereo_depth::preprocess
