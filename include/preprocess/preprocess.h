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
