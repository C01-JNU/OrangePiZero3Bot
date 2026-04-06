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
