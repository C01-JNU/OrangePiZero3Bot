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


#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::Preprocess() : m_censusGpu(nullptr) {
    LOG_INFO("Preprocess 构造函数");
}

// 析构函数在 preprocess_gpu.cpp 中定义（需要 GpuCensusTransform 完整类型）

bool Preprocess::initFromConfig() {
    LOG_INFO("Preprocess::initFromConfig 开始");
    if (!m_denoiser.initFromConfig()) {
        LOG_ERROR("去噪器初始化失败");
        return false;
    }

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    CensusParams censusParams;
    censusParams.window_width = cfg.get<int>("preprocess.census.window_width", 5);
    censusParams.window_height = cfg.get<int>("preprocess.census.window_height", 3);
    censusParams.adaptive_threshold = cfg.get<int>("preprocess.census.adaptive_threshold", 2);
    censusParams.workgroup_size_x = cfg.get<int>("preprocess.gpu.workgroup_size_x", 16);
    censusParams.workgroup_size_y = cfg.get<int>("preprocess.gpu.workgroup_size_y", 16);

    // 读取变换类型
    std::string transformTypeStr = cfg.get<std::string>("preprocess.transform_type", "census");
    if (transformTypeStr == "census") {
        censusParams.transform_type = TransformType::CENSUS;
    } else if (transformTypeStr == "rank") {
        censusParams.transform_type = TransformType::RANK;
    } else {
        LOG_WARN("未知的 transform_type: {}，使用默认 Census", transformTypeStr);
        censusParams.transform_type = TransformType::CENSUS;
    }

    if (!m_censusCpu.init(censusParams)) {
        LOG_ERROR("Census CPU 初始化失败");
        return false;
    }

    m_useGpuConfig = cfg.get<bool>("preprocess.preprocess_use_gpu", false);
    if (m_useGpuConfig) {
        LOG_INFO("配置要求使用 GPU 后端，尝试初始化 GPU 变换模块...");
        if (initGpu()) {
            m_gpuAvailable = true;
            LOG_INFO("GPU 变换模块初始化成功，将使用 GPU 加速");
        } else {
            LOG_WARN("GPU 变换模块初始化失败，将回退到 CPU 后端");
            m_gpuAvailable = false;
        }
    } else {
        LOG_INFO("配置未要求使用 GPU，将使用 CPU 后端");
        m_gpuAvailable = false;
    }

    m_initialized = true;
    LOG_INFO("Preprocess 模块初始化完成");
    return true;
}

bool Preprocess::process(const cv::Mat& left, const cv::Mat& right,
                         cv::Mat& leftCensus, cv::Mat& rightCensus) {
    if (!m_initialized) {
        LOG_ERROR("Preprocess 未初始化");
        return false;
    }
    if (left.empty() || right.empty()) {
        LOG_ERROR("输入图像为空");
        return false;
    }
    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        LOG_ERROR("输入图像必须是 CV_8UC3 彩色图");
        return false;
    }

    if (m_gpuAvailable) {
        return processGpu(left, right, leftCensus, rightCensus);
    } else {
        return processCpu(left, right, leftCensus, rightCensus);
    }
}

bool Preprocess::denoiseOnly(const cv::Mat& src, cv::Mat& dst) {
    return m_denoiser.process(src, dst);
}

bool Preprocess::processCpu(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& leftCensus, cv::Mat& rightCensus) {
    LOG_DEBUG("CPU 处理开始");
    cv::Mat leftGray, rightGray;
    cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);

    cv::Mat leftDenoised, rightDenoised;
    if (!m_denoiser.process(leftGray, leftDenoised) ||
        !m_denoiser.process(rightGray, rightDenoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    if (!m_censusCpu.compute(leftDenoised, rightDenoised, leftCensus, rightCensus)) {
        LOG_ERROR("Census 计算失败");
        return false;
    }
    LOG_DEBUG("CPU 处理完成");
    return true;
}

} // namespace stereo_depth::preprocess
