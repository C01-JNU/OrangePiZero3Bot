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


#include "camera/camera_factory.h"
#include "camera/chusei_camera.h"
#include "camera/mock_camera.h"
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
