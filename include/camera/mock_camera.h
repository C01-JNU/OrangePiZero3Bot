#pragma once

#include "camera_interface.h"
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>

namespace stereo_depth::camera {

/**
 * @brief 模拟摄像头（从测试图像目录读取图片循环播放）
 */
class MockCamera : public CameraInterface {
public:
    MockCamera();
    ~MockCamera() override;

    bool init(int width, int height, int fps) override;
    bool grab(cv::Mat& left, cv::Mat& right) override;
    int getWidth() const override { return m_width; }
    int getHeight() const override { return m_height; }
    std::string getName() const override { return "MockCamera"; }

private:
    void captureThread();

    int m_width = 0;
    int m_height = 0;
    int m_fps = 30;
    int m_single_width = 0;
    std::string m_test_dir;
    std::vector<std::string> m_image_files;
    size_t m_current_index = 0;

    // 线程和队列
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::pair<cv::Mat, cv::Mat>> m_frame_queue;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_initialized{false};
};

} // namespace stereo_depth::camera
