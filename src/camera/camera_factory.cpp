#include "camera/camera_factory.h"
#include "camera/mock_camera.h"
#include "camera/chusei_camera.h"
#include "utils/logger.hpp"

namespace stereo_depth::camera {

CameraPtr CameraFactory::create(const std::string& driver_name) {
    if (driver_name == "mock") {
        LOG_INFO("创建模拟摄像头");
        return std::make_unique<MockCamera>();
    } else if (driver_name == "chusei") {
        LOG_INFO("创建 CHUSEI 摄像头");
        return std::make_unique<ChuseiCamera>();
    }
    LOG_ERROR("不支持的摄像头驱动: {}", driver_name);
    return nullptr;
}

} // namespace stereo_depth::camera
