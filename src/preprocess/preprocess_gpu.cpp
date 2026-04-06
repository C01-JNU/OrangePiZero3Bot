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
#include "preprocess/gpu_census.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::~Preprocess() {
    delete m_censusGpu;
}

bool Preprocess::initGpu() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    if (!cfg.has("preprocess.census.window_width") ||
        !cfg.has("preprocess.census.window_height") ||
        !cfg.has("preprocess.census.adaptive_threshold")) {
        LOG_ERROR("配置缺失: 需要 preprocess.census 相关参数");
        return false;
    }
    int windowWidth = cfg.get<int>("preprocess.census.window_width");
    int windowHeight = cfg.get<int>("preprocess.census.window_height");
    int adaptiveThreshold = cfg.get<int>("preprocess.census.adaptive_threshold");

    // 读取变换类型
    std::string transformTypeStr = cfg.get<std::string>("preprocess.transform_type", "census");
    TransformType type;
    if (transformTypeStr == "census") {
        type = TransformType::CENSUS;
    } else if (transformTypeStr == "rank") {
        type = TransformType::RANK;
    } else {
        LOG_ERROR("未知的变换类型: {}，支持 census/rank", transformTypeStr);
        return false;
    }

    if (windowWidth <= 0 || windowWidth % 2 == 0 || windowHeight <= 0 || windowHeight % 2 == 0) {
        LOG_ERROR("窗口尺寸必须为正奇数");
        return false;
    }

    m_censusGpu = new GpuCensusTransform();
    if (!m_censusGpu->init(windowWidth, windowHeight, adaptiveThreshold, type)) {
        LOG_ERROR("GPU 变换初始化失败");
        delete m_censusGpu;
        m_censusGpu = nullptr;
        return false;
    }
    return true;
}

bool Preprocess::processGpu(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& leftCensus, cv::Mat& rightCensus) {
    if (!m_censusGpu) {
        LOG_ERROR("GPU 变换未初始化");
        return false;
    }

    // CPU 去噪（彩色图转灰度去噪）
    cv::Mat leftGray, rightGray;
    cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);

    cv::Mat leftDenoised, rightDenoised;
    if (!m_denoiser.process(leftGray, leftDenoised) ||
        !m_denoiser.process(rightGray, rightDenoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    // 去噪后的灰度图转换为彩色图（三通道相同）供 GPU 变换使用
    cv::Mat leftDenoisedColor, rightDenoisedColor;
    cv::cvtColor(leftDenoised, leftDenoisedColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rightDenoised, rightDenoisedColor, cv::COLOR_GRAY2BGR);

    if (!m_censusGpu->process(leftDenoisedColor, leftCensus) ||
        !m_censusGpu->process(rightDenoisedColor, rightCensus)) {
        LOG_ERROR("GPU 变换失败");
        return false;
    }
    return true;
}

void* Preprocess::getFilteredImageHandle() const {
    if (m_censusGpu) {
        return m_censusGpu->getOutputImageView();
    }
    return nullptr;
}

int Preprocess::getFilteredImageWidth() const {
    if (m_censusGpu) {
        return m_censusGpu->getWidth();
    }
    return 0;
}

int Preprocess::getFilteredImageHeight() const {
    if (m_censusGpu) {
        return m_censusGpu->getHeight();
    }
    return 0;
}

} // namespace stereo_depth::preprocess
