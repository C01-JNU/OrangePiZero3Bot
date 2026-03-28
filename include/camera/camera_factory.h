#pragma once

#include "camera_interface.h"
#include <memory>
#include <string>

namespace stereo_depth::camera {

class CameraFactory {
public:
    /**
     * @brief 创建摄像头实例
     * @param driver_name 驱动名称（目前支持 "mock" 和 "chusei"）
     * @return 成功返回 CameraPtr，失败返回 nullptr
     */
    static CameraPtr create(const std::string& driver_name);
};

} // namespace stereo_depth::camera
