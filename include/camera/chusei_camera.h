#pragma once

#include "camera_interface.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <thread>

namespace stereo_depth::camera {

class ChuseiCamera : public CameraInterface {
public:
    ChuseiCamera();
    ~ChuseiCamera() override;

    bool init(int width, int height, int fps) override;
    bool grab(cv::Mat& left, cv::Mat& right) override;
    int getWidth() const override { return width_; }
    int getHeight() const override { return height_; }
    std::string getName() const override { return "CHUSEI 3D WebCam"; }

private:
    // 辅助函数
    std::string detectDevice();
    bool verifyDevice(const std::string& device);
    bool runInitScript(const std::string& device);
    bool verifyStereoMode();

    cv::VideoCapture cap_;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 30;
    std::string device_path_;
    std::atomic<bool> initialized_{false};
};

} // namespace stereo_depth::camera
