#pragma once

#include "camera_interface.h"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <string>

namespace stereo_depth::camera {

class ChuseiCamera : public CameraInterface {
public:
    ChuseiCamera();
    ~ChuseiCamera() override;

    bool init(int width, int height, int fps) override;
    bool grab(cv::Mat& left, cv::Mat& right) override;
    int getWidth() const override { return m_width; }
    int getHeight() const override { return m_height; }
    std::string getName() const override { return "CHUSEI 3D WebCam"; }

private:
    std::string detectDevice();
    bool verifyDevice(const std::string& device);
    bool runInitScript(const std::string& device);
    bool verifyStereoMode();

    cv::VideoCapture m_cap;
    int m_width = 0;
    int m_height = 0;
    int m_fps = 30;
    std::string m_devicePath;
    std::atomic<bool> m_initialized{false};
};

} // namespace stereo_depth::camera
