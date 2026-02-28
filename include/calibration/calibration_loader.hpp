#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace stereo_depth {
namespace calibration {

/**
 * @brief 标定参数结构体
 * 
 * 包含双目相机的所有标定参数
 */
struct CalibrationParams {
    // 基本信息
    std::string calibration_date;
    cv::Size image_size;           // 图像尺寸 (320×480)
    int board_width;               // 棋盘格宽度方向内角点数
    int board_height;              // 棋盘格高度方向内角点数
    double square_size;            // 棋盘格方格尺寸 (米)
    int images_used;               // 使用的图像数量
    
    // 标定结果质量
    double rms_error;              // RMS重投影误差
    double baseline_meters;        // 基线长度 (米)
    int best_group_size;           // 最佳分组大小
    int best_group_index;          // 最佳组索引
    
    // 相机内参
    cv::Mat camera_matrix_left;    // 左相机内参矩阵 (3×3)
    cv::Mat camera_matrix_right;   // 右相机内参矩阵 (3×3)
    cv::Mat dist_coeffs_left;      // 左相机畸变系数 (5×1)
    cv::Mat dist_coeffs_right;     // 右相机畸变系数 (5×1)
    
    // 立体外参
    cv::Mat rotation_matrix;       // 旋转矩阵 R (3×3)
    cv::Mat translation_vector;    // 平移向量 T (3×1)
    cv::Mat essential_matrix;      // 本质矩阵 E (3×3)
    cv::Mat fundamental_matrix;    // 基础矩阵 F (3×3)
    
    // 校正参数
    cv::Mat rectification_left;    // 左相机校正变换 R1 (3×3)
    cv::Mat rectification_right;   // 右相机校正变换 R2 (3×3)
    cv::Mat projection_left;       // 左相机投影矩阵 P1 (3×4)
    cv::Mat projection_right;      // 右相机投影矩阵 P2 (3×4)
    cv::Mat disparity_to_depth;    // 视差转深度矩阵 Q (4×4)
    
    /**
     * @brief 检查标定参数是否有效
     * @return 参数有效返回true
     */
    bool isValid() const {
        return !camera_matrix_left.empty() && 
               !camera_matrix_right.empty() &&
               !dist_coeffs_left.empty() &&
               !dist_coeffs_right.empty() &&
               !rectification_left.empty() &&
               !rectification_right.empty() &&
               !projection_left.empty() &&
               !projection_right.empty();
    }
    
    /**
     * @brief 打印标定参数摘要
     */
    void printSummary() const;
};

/**
 * @brief 标定参数加载器
 * 
 * 负责从YAML文件加载和保存标定参数
 */
class CalibrationLoader {
public:
    CalibrationLoader() = default;
    
    /**
     * @brief 从YAML文件加载标定参数
     * @param filepath YAML文件路径
     * @param params 输出参数结构体
     * @return 加载成功返回true
     */
    bool loadFromFile(const std::string& filepath, CalibrationParams& params);
    
    /**
     * @brief 保存标定参数到YAML文件
     * @param filepath YAML文件路径
     * @param params 参数结构体
     * @return 保存成功返回true
     */
    bool saveToFile(const std::string& filepath, const CalibrationParams& params);
    
    /**
     * @brief 检查YAML文件是否存在且可读
     * @param filepath YAML文件路径
     * @return 文件有效返回true
     */
    static bool validateCalibrationFile(const std::string& filepath);
    
private:
    /**
     * @brief 从OpenCV FileStorage读取参数
     * @param fs 已打开的FileStorage
     * @param params 输出参数结构体
     * @return 读取成功返回true
     */
    bool readParamsFromFileStorage(cv::FileStorage& fs, CalibrationParams& params);
    
    /**
     * @brief 写入参数到OpenCV FileStorage
     * @param fs 已打开的FileStorage
     * @param params 参数结构体
     * @return 写入成功返回true
     */
    bool writeParamsToFileStorage(cv::FileStorage& fs, const CalibrationParams& params);
};

} // namespace calibration
} // namespace stereo_depth
