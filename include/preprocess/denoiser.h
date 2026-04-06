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
