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


#pragma once

#include "camera_interface.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

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
    int m_singleWidth = 0;
    std::string m_testDir;
    std::vector<std::string> m_imageFiles;
    size_t m_currentIndex = 0;

    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::pair<cv::Mat, cv::Mat>> m_frameQueue;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_initialized{false};
};

} // namespace stereo_depth::camera
