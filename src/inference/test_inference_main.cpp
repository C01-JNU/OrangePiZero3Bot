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


// test_inference_main.cpp
// 交互式模型转换与测试工具 (中文版)
// 最后更新: 2026-04-06
// 支持 .pt -> ONNX -> MNN 全流程转换
// 路径支持：相对路径（相对于可执行文件目录）和绝对路径，自动去除引号和空白

#include "inference/yolo_detector.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

struct TestConfig {
    std::string input_dir = "images/test";
    std::string output_dir = "images/output";
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png"};
};

TestConfig loadTestConfig() {
    TestConfig cfg;
    auto& global_cfg = stereo_depth::utils::ConfigManager::getInstance().getConfig();
    cfg.input_dir = global_cfg.get<std::string>("inference.test.input_dir", "images/test");
    cfg.output_dir = global_cfg.get<std::string>("inference.test.output_dir", "images/output");
    return cfg;
}

std::string getExecutablePath() {
    char buf[1024];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len != -1) {
        buf[len] = '\0';
        return fs::path(buf).parent_path().string();
    }
    return ".";
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

std::string stripQuotes(const std::string& s) {
    std::string result = trim(s);
    if (result.empty()) return result;
    if (result.front() == '"' || result.front() == '\'')
        result.erase(0, 1);
    if (!result.empty() && (result.back() == '"' || result.back() == '\''))
        result.pop_back();
    return result;
}

std::string resolvePath(const std::string& userInput, const std::string& exeDir) {
    std::string path = stripQuotes(userInput);
    if (path.empty()) return "";
    fs::path p(path);
    if (p.is_absolute()) {
        return p.string();
    } else {
        return (fs::path(exeDir) / p).lexically_normal().string();
    }
}

bool commandExists(const std::string& cmd) {
    return system(("which " + cmd + " > /dev/null 2>&1").c_str()) == 0;
}

bool convertPtToOnnx(const std::string& pt_path, const std::string& onnx_path, int imgsz = 640) {
    if (!fs::exists(pt_path)) {
        std::cerr << "错误: PyTorch 模型文件不存在: " << pt_path << std::endl;
        return false;
    }
    if (!commandExists("yolo")) {
        std::cerr << "错误: 未找到 'yolo' 命令。请先安装 ultralytics: pip install ultralytics" << std::endl;
        return false;
    }
    std::string cmd = "yolo export model=" + pt_path + " format=onnx imgsz=" + std::to_string(imgsz) + " save_dir=" + fs::path(onnx_path).parent_path().string();
    std::cout << "执行命令: " << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "模型转换失败。" << std::endl;
        return false;
    }
    fs::path default_onnx = fs::path(pt_path).stem().string() + ".onnx";
    if (fs::exists(default_onnx) && default_onnx != onnx_path) {
        fs::rename(default_onnx, onnx_path);
    }
    std::cout << "ONNX 模型已保存至: " << onnx_path << std::endl;
    return true;
}

bool convertOnnxToMnn(const std::string& onnx_path, const std::string& mnn_path) {
    if (!fs::exists(onnx_path)) {
        std::cerr << "错误: ONNX 文件不存在: " << onnx_path << std::endl;
        return false;
    }
    std::string mnn_convert_cmd;
    std::string exe_dir = getExecutablePath();
    std::string candidate = exe_dir + "/../MNNConvert";
    if (fs::exists(candidate)) {
        mnn_convert_cmd = candidate;
    } else if (commandExists("MNNConvert")) {
        mnn_convert_cmd = "MNNConvert";
    } else {
        std::cerr << "错误: 未找到 MNNConvert 工具。请确保已编译 MNN 并生成 MNNConvert。" << std::endl;
        return false;
    }
    std::string cmd = mnn_convert_cmd + " -f ONNX --modelFile " + onnx_path +
                      " --MNNModel " + mnn_path + " --bizCode biz";
    std::cout << "执行命令: " << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "模型转换失败。" << std::endl;
        return false;
    }
    std::cout << "MNN 模型已保存至: " << mnn_path << std::endl;
    return true;
}

void convertModelInteractive() {
    std::string exeDir = getExecutablePath();
    std::string input_path_str, output_path_str;
    std::cout << "请输入模型文件路径 (.pt 或 .onnx): ";
    std::cin.ignore();
    std::getline(std::cin, input_path_str);
    std::string input_path = resolvePath(input_path_str, exeDir);
    if (!fs::exists(input_path)) {
        std::cerr << "文件不存在: " << input_path << std::endl;
        return;
    }
    fs::path in_path(input_path);
    std::string ext = in_path.extension().string();
    if (ext != ".pt" && ext != ".onnx") {
        std::cerr << "不支持的文件格式，请提供 .pt 或 .onnx 文件。" << std::endl;
        return;
    }
    std::cout << "请输入输出文件路径 (留空则自动生成): ";
    std::getline(std::cin, output_path_str);
    std::string output_path = resolvePath(output_path_str, exeDir);
    if (output_path.empty()) {
        if (ext == ".pt") {
            output_path = (in_path.parent_path() / (in_path.stem().string() + ".onnx")).string();
        } else {
            output_path = (in_path.parent_path() / (in_path.stem().string() + ".mnn")).string();
        }
    }
    if (ext == ".pt") {
        if (!convertPtToOnnx(input_path, output_path)) {
            std::cerr << "转换失败。" << std::endl;
            return;
        }
        std::string answer;
        std::cout << "是否继续转换为 MNN 格式？(y/n): ";
        std::getline(std::cin, answer);
        if (answer == "y" || answer == "Y") {
            std::string mnn_path_str;
            std::cout << "输入 MNN 输出路径 (默认: " << fs::path(output_path).stem().string() + ".mnn" << "): ";
            std::getline(std::cin, mnn_path_str);
            std::string mnn_path = resolvePath(mnn_path_str, exeDir);
            if (mnn_path.empty()) mnn_path = (fs::path(output_path).parent_path() / (fs::path(output_path).stem().string() + ".mnn")).string();
            convertOnnxToMnn(output_path, mnn_path);
        }
    } else if (ext == ".onnx") {
        convertOnnxToMnn(input_path, output_path);
    }
}

void testModel(const stereo_depth::inference::YOLOConfig& model_cfg,
               const TestConfig& test_cfg,
               const std::string& exe_dir) {
    fs::path input_path = fs::path(exe_dir) / test_cfg.input_dir;
    fs::path output_path = fs::path(exe_dir) / test_cfg.output_dir;
    if (!fs::exists(input_path)) {
        std::cerr << "输入目录不存在: " << input_path << std::endl;
        return;
    }
    fs::create_directories(output_path);

    std::vector<fs::path> images;
    for (const auto& ext : test_cfg.extensions) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ext) {
                images.push_back(entry.path());
            }
        }
    }
    if (images.empty()) {
        std::cout << "在 " << input_path << " 中没有找到图像文件。" << std::endl;
        return;
    }
    std::sort(images.begin(), images.end());

    stereo_depth::inference::YOLODetector detector;
    if (!detector.init(model_cfg)) {
        std::cerr << "检测器初始化失败。" << std::endl;
        return;
    }

    double total_time = 0.0;
    int successful = 0;
    std::vector<double> frame_times;

    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = cv::imread(images[i].string());
        if (img.empty()) {
            std::cerr << "无法读取图像: " << images[i].string() << std::endl;
            continue;
        }
        std::vector<stereo_depth::inference::DetectionBox> boxes;
        auto start = std::chrono::steady_clock::now();
        bool ok = detector.detect(img, boxes);
        auto end = std::chrono::steady_clock::now();
        if (!ok) {
            std::cerr << "检测失败: " << images[i].string() << std::endl;
            continue;
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        frame_times.push_back(elapsed_ms);
        total_time += elapsed_ms;
        successful++;

        cv::Mat vis = img.clone();
        for (const auto& box : boxes) {
            int x1 = static_cast<int>(box.x1 * img.cols);
            int y1 = static_cast<int>(box.y1 * img.rows);
            int x2 = static_cast<int>(box.x2 * img.cols);
            int y2 = static_cast<int>(box.y2 * img.rows);
            cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            std::string label = box.class_name + ":" + std::to_string(box.confidence).substr(0,4);
            cv::putText(vis, label, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }
        fs::path out_file = output_path / (images[i].stem().string() + "_out.jpg");
        cv::imwrite(out_file.string(), vis);
        std::cout << "已处理 " << images[i].filename() << " -> " << out_file.filename()
                  << " (检测到 " << boxes.size() << " 个目标, 耗时 " << std::fixed << std::setprecision(2)
                  << elapsed_ms << " ms)" << std::endl;
    }

    if (successful == 0) {
        std::cout << "没有成功检测的图像。" << std::endl;
        return;
    }

    double avg_ms = total_time / successful;
    double fps = 1000.0 / avg_ms;
    std::cout << "\n========== 统计信息 ==========" << std::endl;
    std::cout << "总图像数: " << images.size() << std::endl;
    std::cout << "成功处理: " << successful << std::endl;
    std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_ms << " ms" << std::endl;
    std::cout << "平均帧率: " << std::fixed << std::setprecision(2) << fps << " FPS" << std::endl;
    std::cout << "结果保存在: " << output_path << std::endl;
}

int main() {
    stereo_depth::utils::Logger::initialize("test_inference", spdlog::level::info);

    std::string config_path = getExecutablePath() + "/config";
    if (!stereo_depth::utils::ConfigManager::getInstance().loadGlobalConfig(config_path)) {
        std::cerr << "警告: 无法从 " << config_path << " 加载配置，将使用默认值。" << std::endl;
    }

    stereo_depth::inference::YOLOConfig model_cfg;
    auto& cfg = stereo_depth::utils::ConfigManager::getInstance().getConfig();
    model_cfg.model_path = cfg.get<std::string>("inference.model.path", "models/yolo26n.mnn");
    model_cfg.input_width = cfg.get<int>("inference.model.input_width", 0);
    model_cfg.input_height = cfg.get<int>("inference.model.input_height", 0);
    model_cfg.confidence_threshold = cfg.get<float>("inference.confidence_threshold", 0.25f);
    model_cfg.nms_threshold = cfg.get<float>("inference.nms_threshold", 0.45f);
    model_cfg.num_classes = cfg.get<int>("inference.num_classes", 80);
    model_cfg.use_gpu = cfg.get<bool>("inference.use_gpu", false);
    model_cfg.letterbox = cfg.get<bool>("inference.letterbox", true);

    TestConfig test_cfg = loadTestConfig();
    std::string exe_dir = getExecutablePath();

    int choice = -1;
    while (choice != 0) {
        std::cout << "\n===== 模型推理测试工具 =====" << std::endl;
        std::cout << "1. 转换模型格式 (.pt -> ONNX 或 ONNX -> MNN)" << std::endl;
        std::cout << "2. 批量测试模型（从目录读取图像）" << std::endl;
        std::cout << "0. 退出" << std::endl;
        std::cout << "请输入选项: ";
        std::cin >> choice;
        std::cin.ignore();

        if (choice == 1) {
            convertModelInteractive();
        } else if (choice == 2) {
            testModel(model_cfg, test_cfg, exe_dir);
        } else if (choice == 0) {
            std::cout << "退出程序。" << std::endl;
        } else {
            std::cout << "无效选项，请重新输入。" << std::endl;
        }
    }
    return 0;
}
