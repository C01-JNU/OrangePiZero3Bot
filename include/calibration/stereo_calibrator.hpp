#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace stereo_depth {
namespace calibration {

/**
 * @brief 立体相机标定器（简化版，带验证图像生成）
 * 
 * 使用所有有效图像进行标定，不进行分组优化。
 * 从配置文件读取参数，自动匹配左右图像对。
 * 生成灰度/彩色校正验证图像，绘制水平线和有效区域。
 */
class StereoCalibrator {
public:
    StereoCalibrator() = default;
    ~StereoCalibrator() = default;

    // 禁止拷贝
    StereoCalibrator(const StereoCalibrator&) = delete;
    StereoCalibrator& operator=(const StereoCalibrator&) = delete;

    /**
     * @brief 设置基础路径（用于定位图像和输出目录）
     */
    void setBasePath(const std::string& path) { m_basePath = path; }

    /**
     * @brief 执行立体标定
     * @return 是否标定成功
     */
    bool calibrate();

    /**
     * @brief 获取标定误差（RMS）
     */
    double getCalibrationError() const { return m_rmsError; }

    /**
     * @brief 获取使用的图像数量
     */
    int getImagesUsed() const { return m_imagesUsed; }

private:
    /**
     * @brief 加载配置参数
     */
    bool loadConfiguration();

    /**
     * @brief 查找标定图像对
     * @return 左图路径和右图路径的 pair 列表
     */
    std::vector<std::pair<std::string, std::string>> findCalibrationImagePairs();

    /**
     * @brief 检测棋盘格角点
     * @param image 输入图像
     * @param corners 输出的角点坐标
     * @return 是否检测成功
     */
    bool detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);

    /**
     * @brief 执行标定
     * @param objectPoints 物体点（所有图像）
     * @param imagePointsLeft 左图像点
     * @param imagePointsRight 右图像点
     * @return 是否标定成功
     */
    bool performCalibration(
        const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePointsLeft,
        const std::vector<std::vector<cv::Point2f>>& imagePointsRight);

    /**
     * @brief 保存标定结果到YAML文件
     */
    bool saveCalibrationResults();

    /**
     * @brief 生成标定报告（文本文件，包含详细参数）
     */
    bool generateCalibrationReport();

    /**
     * @brief 验证标定结果并生成校正图像
     */
    bool validateCalibrationResults();

private:
    // 标定参数
    cv::Size m_boardSize;            // 棋盘格内角点数
    float m_squareSize = 0.0f;       // 方格物理尺寸（米）
    cv::Size m_imageSize;            // 图像尺寸

    // 标定结果
    cv::Mat m_cameraMatrixLeft;
    cv::Mat m_distCoeffsLeft;
    cv::Mat m_cameraMatrixRight;
    cv::Mat m_distCoeffsRight;
    cv::Mat m_rotationMatrix;
    cv::Mat m_translationVector;
    cv::Mat m_essentialMatrix;
    cv::Mat m_fundamentalMatrix;
    cv::Mat m_rectificationTransformLeft;
    cv::Mat m_rectificationTransformRight;
    cv::Mat m_projectionMatrixLeft;
    cv::Mat m_projectionMatrixRight;
    cv::Mat m_disparityToDepthMappingMatrix;

    // 校正映射
    cv::Mat m_leftMap1, m_leftMap2;
    cv::Mat m_rightMap1, m_rightMap2;

    // 有效 ROI（立体校正后的有效区域）
    cv::Rect m_validRoiLeft;
    cv::Rect m_validRoiRight;

    double m_rmsError = 0.0;
    int m_imagesUsed = 0;

    // 路径配置
    std::string m_calibrationDir;
    std::string m_outputDir;
    std::string m_basePath;          // 基础路径，用于构建绝对路径
};

} // namespace calibration
} // namespace stereo_depth
