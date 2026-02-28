#include "calibration/calibration_loader.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "utils/logger.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <unistd.h>
#include <limits.h>

namespace fs = std::filesystem;

std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::filesystem::path exePath(result);
        return exePath.parent_path().string();
    }
    return ".";
}

int main(int argc, char** argv) {
    std::string exeDir = getExeDir();
    
    // 命令行参数解析（如果指定绝对路径则使用，否则基于 exeDir 构建）
    std::string calibration_file = (argc >= 2) ? argv[1] : (exeDir + "/calibration_results/stereo_calibration.yml");
    std::string test_image_dir   = (argc >= 3) ? argv[2] : (exeDir + "/images/test");
    std::string output_dir        = (argc >= 4) ? argv[3] : (exeDir + "/images/calibrated");
    std::string mode_str          = (argc >= 5) ? argv[4] : "scale_to_fit";
    bool save_comparison = (argc >= 6) ? (std::string(argv[5]) == "true") : true;
    
    // 解析校正模式
    stereo_depth::calibration::RectificationMode mode;
    if (mode_str == "raw") {
        mode = stereo_depth::calibration::RectificationMode::RAW;
    } else if (mode_str == "crop_only") {
        mode = stereo_depth::calibration::RectificationMode::CROP_ONLY;
    } else {
        mode = stereo_depth::calibration::RectificationMode::SCALE_TO_FIT;
    }
    
    // 显示帮助
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "立体校正测试程序\n";
        std::cout << "用法: " << argv[0] << " [calibration_file] [test_image_dir] [output_dir] [mode] [save_comparison]\n";
        std::cout << "参数:\n";
        std::cout << "  calibration_file: 标定YAML文件路径 (默认: " << exeDir << "/calibration_results/stereo_calibration.yml)\n";
        std::cout << "  test_image_dir: 测试图像目录 (默认: " << exeDir << "/images/test)\n";
        std::cout << "  output_dir: 输出目录 (默认: " << exeDir << "/images/calibrated)\n";
        std::cout << "  mode: 校正模式 - raw, crop_only, scale_to_fit (默认: scale_to_fit)\n";
        std::cout << "  save_comparison: 是否保存对比图 - true/false (默认: true)\n";
        return 0;
    }
    
    LOG_INFO("=== 立体校正测试程序 (方案2) ===");
    LOG_INFO("标定文件: {}", calibration_file);
    LOG_INFO("测试图像目录: {}", test_image_dir);
    LOG_INFO("输出目录: {}", output_dir);
    LOG_INFO("校正模式: {}", mode_str);
    LOG_INFO("保存对比图: {}", save_comparison ? "是" : "否");
    
    // 1. 初始化立体校正器
    stereo_depth::calibration::StereoRectifier rectifier;
    if (!rectifier.loadAndInitialize(calibration_file, mode)) {
        LOG_ERROR("初始化立体校正器失败");
        return -1;
    }
    
    // 2. 显示缩放信息（如果使用SCALE_TO_FIT模式）
    if (mode == stereo_depth::calibration::RectificationMode::SCALE_TO_FIT) {
        auto scale_info = rectifier.getScaleInfo();
        LOG_INFO("缩放信息:");
        LOG_INFO("  原始ROI: {}x{} (在原始图像中的位置: {}, {})",
                 scale_info.roi.width, scale_info.roi.height,
                 scale_info.roi.x, scale_info.roi.y);
        LOG_INFO("  缩放因子: {:.3f}", scale_info.scale_factor);
        LOG_INFO("  缩放后尺寸: {}x{}", scale_info.scaled_size.width, scale_info.scaled_size.height);
        LOG_INFO("  填充偏移: ({}, {})", scale_info.offset.x, scale_info.offset.y);
        LOG_INFO("  有效像素比例: {:.1f}%", scale_info.effective_ratio * 100.0);
    }
    
    // 3. 获取测试图像
    std::vector<std::string> test_images;
    try {
        for (const auto& entry : fs::directory_iterator(test_image_dir)) {
            if (entry.is_regular_file() && 
                (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
                test_images.push_back(entry.path().string());
            }
        }
        std::sort(test_images.begin(), test_images.end());
        
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("无法访问测试目录 {}: {}", test_image_dir, e.what());
        return -1;
    }
    
    if (test_images.empty()) {
        LOG_ERROR("未找到测试图像");
        return -1;
    }
    
    LOG_INFO("找到 {} 张测试图像", test_images.size());
    
    // 4. 创建输出目录
    fs::create_directories(output_dir);
    fs::create_directories(output_dir + "/left");
    fs::create_directories(output_dir + "/right");
    fs::create_directories(output_dir + "/comparison");
    
    // 5. 处理每张图像
    int processed_count = 0;
    
    for (size_t i = 0; i < std::min(test_images.size(), static_cast<size_t>(10)); ++i) {
        const auto& image_path = test_images[i];
        
        LOG_INFO("处理图像 {}/{}: {}", i + 1, test_images.size(), fs::path(image_path).filename().string());
        
        try {
            // 读取拼接图像
            cv::Mat stitched_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (stitched_image.empty()) {
                LOG_ERROR("无法读取图像: {}", image_path);
                continue;
            }
            
            // 分割为左右眼
            int single_width = stitched_image.cols / 2;
            int height = stitched_image.rows;
            
            cv::Mat left_raw = stitched_image(cv::Rect(0, 0, single_width, height)).clone();
            cv::Mat right_raw = stitched_image(cv::Rect(single_width, 0, single_width, height)).clone();
            
            // 检查图像尺寸
            cv::Size expected_size = rectifier.getImageSize();
            if (left_raw.size() != expected_size) {
                LOG_WARN("调整图像尺寸从 {}x{} 到 {}x{}",
                         left_raw.cols, left_raw.rows,
                         expected_size.width, expected_size.height);
                cv::resize(left_raw, left_raw, expected_size);
                cv::resize(right_raw, right_raw, expected_size);
            }
            
            // 执行立体校正（使用指定模式）
            cv::Mat left_rectified, right_rectified;
            if (!rectifier.rectifyPair(left_raw, right_raw, 
                                       left_rectified, right_rectified,
                                       mode)) {
                LOG_ERROR("校正图像失败");
                continue;
            }
            
            // 保存校正结果
            std::string filename_base = fs::path(image_path).stem().string();
            
            // 保存左眼图像
            std::string left_path = output_dir + "/left/" + filename_base + "_left.png";
            cv::imwrite(left_path, left_rectified);
            
            // 保存右眼图像
            std::string right_path = output_dir + "/right/" + filename_base + "_right.png";
            cv::imwrite(right_path, right_rectified);
            
            // 保存对比图（原始 vs 校正后）
            if (save_comparison) {
                // 创建对比图：原始左图 | 校正后左图 | 原始右图 | 校正后右图
                cv::Mat comparison = cv::Mat::zeros(
                    std::max(left_raw.rows, left_rectified.rows) * 2,
                    left_raw.cols + left_rectified.cols,
                    CV_8UC1
                );
                
                // 第一行：原始左右图
                left_raw.copyTo(comparison(cv::Rect(0, 0, left_raw.cols, left_raw.rows)));
                right_raw.copyTo(comparison(cv::Rect(left_raw.cols, 0, right_raw.cols, right_raw.rows)));
                
                // 第二行：校正后左右图
                left_rectified.copyTo(comparison(cv::Rect(0, left_raw.rows, left_rectified.cols, left_rectified.rows)));
                right_rectified.copyTo(comparison(cv::Rect(left_rectified.cols, right_raw.rows, right_rectified.cols, right_rectified.rows)));
                
                // 添加标签
                cv::putText(comparison, "Original Left", cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
                cv::putText(comparison, "Original Right", cv::Point(left_raw.cols + 10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
                cv::putText(comparison, "Rectified Left", cv::Point(10, left_raw.rows + 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
                cv::putText(comparison, "Rectified Right", cv::Point(left_rectified.cols + 10, right_raw.rows + 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255), 2);
                
                std::string comparison_path = output_dir + "/comparison/" + filename_base + "_comparison.png";
                cv::imwrite(comparison_path, comparison);
            }
            
            processed_count++;
            
        } catch (const std::exception& e) {
            LOG_ERROR("处理图像时发生异常: {}", e.what());
        }
    }
    
    // 6. 输出统计信息
    LOG_INFO("=== 处理完成 ===");
    LOG_INFO("总图像数: {}", test_images.size());
    LOG_INFO("成功处理: {}", processed_count);
    LOG_INFO("校正结果保存在: {}", output_dir);
    LOG_INFO("左眼图像: {}/left/", output_dir);
    LOG_INFO("右眼图像: {}/right/", output_dir);
    
    if (save_comparison) {
        LOG_INFO("对比图: {}/comparison/", output_dir);
    }
    
    return 0;
}
