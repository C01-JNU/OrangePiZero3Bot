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


// yolo_detector.h
// YOLO 目标检测器（支持动态输入尺寸 + Letterbox）
// 最后更新: 2026-04-06
// 作者: C01-JNU

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

namespace MNN {
class Interpreter;
class Session;
class Tensor;
} // namespace MNN

namespace stereo_depth::inference {

struct DetectionBox {
    float x1, y1, x2, y2;   // 归一化坐标 (0~1)
    float confidence;
    int class_id;
    std::string class_name;
};

struct YOLOConfig {
    std::string model_path;
    int input_width = 0;          // 0 表示自动从第一帧获取
    int input_height = 0;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int num_classes = 80;
    bool use_gpu = false;
    bool letterbox = true;        // 是否保持宽高比加灰边
};

class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector();

    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;
    YOLODetector(YOLODetector&&) = default;
    YOLODetector& operator=(YOLODetector&&) = default;

    bool initFromConfig();
    bool init(const YOLOConfig& config);

    // CPU 推理（自动适应第一帧尺寸）
    bool detect(const cv::Mat& image, std::vector<DetectionBox>& results);

    // GPU 预留接口
    bool detectFromGpuImage(void* vkImageView, int width, int height,
                            std::vector<DetectionBox>& results);
    void setVulkanResources(void* device, void* queue, void* commandPool);

    bool isInitialized() const { return m_initialized; }
    const YOLOConfig& getConfig() const { return m_config; }

private:
    bool loadModel(int input_width, int input_height);
    bool preprocess(const cv::Mat& image, MNN::Tensor* input_tensor,
                    float& scale_x, float& scale_y, int& pad_left, int& pad_top);
    bool postprocess(const std::vector<MNN::Tensor*>& output_tensors,
                     std::vector<DetectionBox>& results,
                     float scale_x, float scale_y, int pad_left, int pad_top);
    void applyNMS(std::vector<DetectionBox>& boxes, float iou_threshold);

private:
    YOLOConfig m_config;
    std::unique_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session* m_session = nullptr;
    bool m_initialized = false;
    bool m_input_size_fixed = false;   // 是否已确定输入尺寸

    void* m_vk_device = nullptr;
    void* m_vk_queue = nullptr;
    void* m_vk_command_pool = nullptr;
};

} // namespace stereo_depth::inference
