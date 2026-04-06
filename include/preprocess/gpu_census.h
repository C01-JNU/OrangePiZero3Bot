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


// gpu_census.h
// Census / Rank 变换 GPU 实现 (Vulkan) - 支持自适应阈值
// 最后更新: 2026-04-06
// 作者: C01-JNU & AI助手

#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include "census.h"   // 引入 TransformType 枚举

namespace stereo_depth::preprocess {

/**
 * @brief GPU 实现的 Census / Rank 变换类
 */
class GpuCensusTransform {
public:
    GpuCensusTransform();
    ~GpuCensusTransform();

    /**
     * @brief 初始化 Vulkan 资源，并选择变换类型
     * @param windowWidth  窗口宽度（奇数）
     * @param windowHeight 窗口高度（奇数）
     * @param adaptiveThreshold 自适应阈值（0 表示禁用）
     * @param type 变换类型（CENSUS 或 RANK）
     * @return 成功返回 true
     */
    bool init(int windowWidth, int windowHeight, int adaptiveThreshold, TransformType type);

    /**
     * @brief 对 BGR 图像执行变换
     * @param inputBgr 输入图像 (CV_8UC3)
     * @param output 输出变换图 (CV_16U)
     * @return 成功返回 true
     */
    bool process(const cv::Mat& inputBgr, cv::Mat& output);

    /**
     * @brief 获取输出图像的 Vulkan 图像视图句柄
     * @return VkImageView 转换为 void*
     */
    void* getOutputImageView() const;

    /**
     * @brief 获取输入图像的 Vulkan 图像视图句柄（去噪后的彩色图）
     * @return VkImageView 转换为 void*
     */
    void* getInputImageView() const;

    /**
     * @brief 获取当前输出图像的宽度
     */
    int getWidth() const;

    /**
     * @brief 获取当前输出图像的高度
     */
    int getHeight() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace stereo_depth::preprocess
