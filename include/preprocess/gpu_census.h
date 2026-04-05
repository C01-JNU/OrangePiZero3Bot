// gpu_census.h
// Census 变换 GPU 实现 (Vulkan) - 动态创建资源，无硬编码
// 最后更新: 2026-04-05

#pragma once

#include <opencv2/core.hpp>
#include <memory>

namespace stereo_depth::preprocess {

/**
 * @brief GPU Census 变换类
 */
class GpuCensusTransform {
public:
    GpuCensusTransform();
    ~GpuCensusTransform();

    /**
     * @brief 初始化 Vulkan 实例、设备、管线等（不创建图像资源）
     * @param windowWidth  窗口宽度（奇数）
     * @param windowHeight 窗口高度（奇数）
     * @param adaptiveThreshold 自适应阈值
     * @return 成功返回 true
     */
    bool init(int windowWidth, int windowHeight, int adaptiveThreshold);

    /**
     * @brief 对 BGR 图像执行 Census 变换（动态创建/重建资源）
     * @param inputBgr 输入图像 (CV_8UC3)
     * @param outputCensus 输出 Census 图 (CV_16U)
     * @return 成功返回 true
     */
    bool process(const cv::Mat& inputBgr, cv::Mat& outputCensus);

    /**
     * @brief 获取输出图像的 Vulkan 图像视图句柄
     * @return VkImageView 转换为 void*
     */
    void* getOutputImageView() const;

    int getWidth() const;
    int getHeight() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace stereo_depth::preprocess
