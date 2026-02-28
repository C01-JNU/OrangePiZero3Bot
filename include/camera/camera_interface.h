#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <memory>

namespace stereo_depth::camera {

/**
 * @brief 摄像头通用抽象接口
 */
class CameraInterface {
public:
    virtual ~CameraInterface() = default;

    /**
     * @brief 初始化摄像头
     * @param width 请求的图像宽度
     * @param height 请求的图像高度
     * @param fps 目标帧率
     * @return 成功返回 true
     */
    virtual bool init(int width, int height, int fps) = 0;

    /**
     * @brief 获取一帧双目图像（阻塞，直到成功或超时）
     * @param left 输出左图（CV_8UC1）
     * @param right 输出右图（CV_8UC1）
     * @return 成功返回 true
     */
    virtual bool grab(cv::Mat& left, cv::Mat& right) = 0;

    /**
     * @brief 获取实际宽度
     */
    virtual int getWidth() const = 0;

    /**
     * @brief 获取实际高度
     */
    virtual int getHeight() const = 0;

    /**
     * @brief 获取摄像头型号名称
     */
    virtual std::string getName() const = 0;
};

using CameraPtr = std::unique_ptr<CameraInterface>;

} // namespace stereo_depth::camera
