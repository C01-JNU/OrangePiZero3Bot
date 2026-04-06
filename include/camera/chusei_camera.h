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
