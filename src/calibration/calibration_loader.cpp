#include "calibration/calibration_loader.hpp"
#include "utils/logger.hpp"
#include <sys/stat.h>
#include <iomanip>
#include <sstream>

namespace stereo_depth {
namespace calibration {

bool fileExists(const std::string& filepath) {
    struct stat buffer;
    return (stat(filepath.c_str(), &buffer) == 0);
}

void CalibrationParams::printSummary() const {
    LOG_INFO("标定参数摘要:");
    LOG_INFO("  标定日期: {}", calibration_date);
    LOG_INFO("  图像尺寸: {}x{}", image_size.width, image_size.height);
    LOG_INFO("  棋盘格: {}x{} 内角点", board_width, board_height);
    LOG_INFO("  方格尺寸: {:.2f} mm", square_size * 1000.0);
    LOG_INFO("  使用图像数: {}", images_used);
    LOG_INFO("  RMS误差: {:.4f} 像素", rms_error);
    LOG_INFO("  基线长度: {:.2f} mm", baseline_meters * 1000.0);
    LOG_INFO("  最佳分组: 大小={}, 索引={}", best_group_size, best_group_index);
    
    LOG_INFO("  左相机内参:");
    LOG_INFO("    fx={:.2f}, fy={:.2f}", camera_matrix_left.at<double>(0, 0), 
                                        camera_matrix_left.at<double>(1, 1));
    LOG_INFO("    cx={:.2f}, cy={:.2f}", camera_matrix_left.at<double>(0, 2), 
                                        camera_matrix_left.at<double>(1, 2));
    
    LOG_INFO("  右相机内参:");
    LOG_INFO("    fx={:.2f}, fy={:.2f}", camera_matrix_right.at<double>(0, 0), 
                                        camera_matrix_right.at<double>(1, 1));
    LOG_INFO("    cx={:.2f}, cy={:.2f}", camera_matrix_right.at<double>(0, 2), 
                                        camera_matrix_right.at<double>(1, 2));
}

bool CalibrationLoader::loadFromFile(const std::string& filepath, CalibrationParams& params) {
    if (!validateCalibrationFile(filepath)) {
        LOG_ERROR("标定文件无效: {}", filepath);
        return false;
    }
    
    LOG_INFO("正在加载标定参数: {}", filepath);
    
    try {
        cv::FileStorage fs(filepath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            LOG_ERROR("无法打开标定文件: {}", filepath);
            return false;
        }
        
        return readParamsFromFileStorage(fs, params);
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV异常加载标定文件: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("标准异常加载标定文件: {}", e.what());
        return false;
    }
}

bool CalibrationLoader::saveToFile(const std::string& filepath, const CalibrationParams& params) {
    LOG_INFO("正在保存标定参数到: {}", filepath);
    
    try {
        cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            LOG_ERROR("无法创建标定文件: {}", filepath);
            return false;
        }
        
        return writeParamsToFileStorage(fs, params);
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV异常保存标定文件: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("标准异常保存标定文件: {}", e.what());
        return false;
    }
}

bool CalibrationLoader::validateCalibrationFile(const std::string& filepath) {
    if (!fileExists(filepath)) {
        LOG_ERROR("标定文件不存在: {}", filepath);
        return false;
    }
    
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        LOG_ERROR("标定文件无法打开或格式错误: {}", filepath);
        return false;
    }
    
    if (!fs["camera_matrix_left"].isNone() && 
        !fs["camera_matrix_right"].isNone() &&
        !fs["distortion_coefficients_left"].isNone() &&
        !fs["distortion_coefficients_right"].isNone()) {
        LOG_DEBUG("标定文件验证通过: {}", filepath);
        return true;
    }
    
    LOG_ERROR("标定文件缺少必需字段: {}", filepath);
    return false;
}

bool CalibrationLoader::readParamsFromFileStorage(cv::FileStorage& fs, CalibrationParams& params) {
    fs["calibration_date"] >> params.calibration_date;
    
    int image_width = 0, image_height = 0;
    fs["image_width"] >> image_width;
    fs["image_height"] >> image_height;
    params.image_size = cv::Size(image_width, image_height);
    
    fs["board_width"] >> params.board_width;
    fs["board_height"] >> params.board_height;
    fs["square_size"] >> params.square_size;
    fs["images_used"] >> params.images_used;
    fs["best_group_size"] >> params.best_group_size;
    fs["best_group_index"] >> params.best_group_index;
    fs["rms_error"] >> params.rms_error;
    // 读取基线长度，若文件不存在则保持默认0
    fs["baseline_meters"] >> params.baseline_meters;
    
    fs["camera_matrix_left"] >> params.camera_matrix_left;
    fs["camera_matrix_right"] >> params.camera_matrix_right;
    fs["distortion_coefficients_left"] >> params.dist_coeffs_left;
    fs["distortion_coefficients_right"] >> params.dist_coeffs_right;
    
    fs["rotation_matrix"] >> params.rotation_matrix;
    fs["translation_vector"] >> params.translation_vector;
    fs["essential_matrix"] >> params.essential_matrix;
    fs["fundamental_matrix"] >> params.fundamental_matrix;
    
    fs["rectification_transform_left"] >> params.rectification_left;
    fs["rectification_transform_right"] >> params.rectification_right;
    fs["projection_matrix_left"] >> params.projection_left;
    fs["projection_matrix_right"] >> params.projection_right;
    fs["disparity_to_depth_mapping_matrix"] >> params.disparity_to_depth;
    
    if (!params.isValid()) {
        LOG_ERROR("从文件读取的标定参数不完整");
        return false;
    }
    
    LOG_INFO("成功加载标定参数");
    params.printSummary();
    
    return true;
}

bool CalibrationLoader::writeParamsToFileStorage(cv::FileStorage& fs, const CalibrationParams& params) {
    fs << "calibration_date" << params.calibration_date;
    fs << "image_width" << params.image_size.width;
    fs << "image_height" << params.image_size.height;
    fs << "board_width" << params.board_width;
    fs << "board_height" << params.board_height;
    fs << "square_size" << params.square_size;
    fs << "images_used" << params.images_used;
    fs << "best_group_size" << params.best_group_size;
    fs << "best_group_index" << params.best_group_index;
    fs << "rms_error" << params.rms_error;
    fs << "baseline_meters" << params.baseline_meters;
    
    fs << "camera_matrix_left" << params.camera_matrix_left;
    fs << "camera_matrix_right" << params.camera_matrix_right;
    fs << "distortion_coefficients_left" << params.dist_coeffs_left;
    fs << "distortion_coefficients_right" << params.dist_coeffs_right;
    
    fs << "rotation_matrix" << params.rotation_matrix;
    fs << "translation_vector" << params.translation_vector;
    fs << "essential_matrix" << params.essential_matrix;
    fs << "fundamental_matrix" << params.fundamental_matrix;
    
    fs << "rectification_transform_left" << params.rectification_left;
    fs << "rectification_transform_right" << params.rectification_right;
    fs << "projection_matrix_left" << params.projection_left;
    fs << "projection_matrix_right" << params.projection_right;
    fs << "disparity_to_depth_mapping_matrix" << params.disparity_to_depth;
    
    LOG_INFO("成功保存标定参数到文件");
    return true;
}

} // namespace calibration
} // namespace stereo_depth
