// gpu_bilateral.h
// 双边滤波 GPU 实现 - 使用外部 Vulkan 资源
// 最后更新: 2026-04-05
// 作者: C01-JNU

#pragma once

#include <opencv2/core.hpp>
#include <memory>

typedef struct VkDevice_T* VkDevice;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkQueue_T* VkQueue;
typedef struct VkCommandPool_T* VkCommandPool;

namespace stereo_depth::preprocess {

/**
 * @brief GPU 双边滤波类
 */
class GpuBilateralFilter {
public:
    GpuBilateralFilter();
    ~GpuBilateralFilter();

    /**
     * @brief 初始化模块（使用外部 Vulkan 资源）
     * @param diameter 滤波直径（奇数）
     * @param sigmaColor 颜色空间标准差
     * @param sigmaSpace 空间标准差
     * @param physicalDevice Vulkan 物理设备句柄
     * @param device Vulkan 设备句柄
     * @param queue 计算队列句柄
     * @param commandPool 命令池句柄
     * @return 成功返回 true
     */
    bool init(int diameter, float sigmaColor, float sigmaSpace,
              VkPhysicalDevice physicalDevice, VkDevice device,
              VkQueue queue, VkCommandPool commandPool);

    /**
     * @brief 对 BGR 图像执行双边滤波
     * @param inputBgr 输入图像 (CV_8UC3)
     * @param outputBgr 输出图像 (CV_8UC3)
     * @return 成功返回 true
     */
    bool process(const cv::Mat& inputBgr, cv::Mat& outputBgr);

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
