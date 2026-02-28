#pragma once

#include "calibration/calibration_loader.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <utility>

namespace stereo_depth {
namespace calibration {

// 输出模式枚举
enum class RectificationMode {
    RAW,            // 原始校正，不裁剪
    CROP_ONLY,      // 只裁剪有效区域
    SCALE_TO_FIT    // 裁剪+等比缩放+填充到原始尺寸 (方案2)
};

/**
 * @brief 立体校正处理器
 * 
 * 负责使用标定参数对双目图像进行立体校正
 */
class StereoRectifier {
public:
    StereoRectifier();
    
    /**
     * @brief 构造函数，直接加载标定文件
     * @param calibration_file 标定文件路径
     * @param mode 校正输出模式
     */
    explicit StereoRectifier(const std::string& calibration_file, 
                           RectificationMode mode = RectificationMode::SCALE_TO_FIT);
    
    ~StereoRectifier();
    
    // 禁止拷贝，允许移动
    StereoRectifier(const StereoRectifier&) = delete;
    StereoRectifier& operator=(const StereoRectifier&) = delete;
    StereoRectifier(StereoRectifier&&) = default;
    StereoRectifier& operator=(StereoRectifier&&) = default;
    
    /**
     * @brief 使用标定参数初始化校正器
     * @param params 标定参数
     * @param mode 校正输出模式
     * @return 初始化成功返回true
     */
    bool initialize(const CalibrationParams& params, 
                   RectificationMode mode = RectificationMode::SCALE_TO_FIT);
    
    /**
     * @brief 从文件加载标定参数并初始化
     * @param calibration_file 标定文件路径
     * @return 初始化成功返回true
     */
    bool loadAndInitialize(const std::string& calibration_file);
    
    /**
     * @brief 从文件加载标定参数并初始化，指定模式
     * @param calibration_file 标定文件路径
     * @param mode 校正模式
     */
    bool loadAndInitialize(const std::string& calibration_file, RectificationMode mode);
    
    /**
     * @brief 校正一对立体图像
     * @param left_image 左眼原始图像 (单通道灰度图)
     * @param right_image 右眼原始图像 (单通道灰度图)
     * @param left_rectified 输出：校正后的左眼图像
     * @param right_rectified 输出：校正后的右眼图像
     * @param mode 校正模式（覆盖构造函数中的模式）
     * @return 校正成功返回true
     */
    bool rectifyPair(const cv::Mat& left_image, 
                     const cv::Mat& right_image,
                     cv::Mat& left_rectified,
                     cv::Mat& right_rectified,
                     RectificationMode mode = RectificationMode::SCALE_TO_FIT);
    
    /**
     * @brief 校正一批立体图像
     * @param image_pairs 输入图像对列表
     * @param rectified_pairs 输出校正后图像对列表
     * @param crop_to_valid_roi 兼容旧接口，实际忽略，通过模式控制
     * @return 校正成功返回true
     */
    bool rectifyBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                     std::vector<std::pair<cv::Mat, cv::Mat>>& rectified_pairs,
                     bool crop_to_valid_roi = false);
    
    /**
     * @brief 获取标定参数
     */
    const CalibrationParams& getCalibrationParams() const;
    
    /**
     * @brief 获取左相机校正映射（当前模式）
     */
    std::pair<cv::Mat, cv::Mat> getLeftMaps() const;
    
    /**
     * @brief 获取右相机校正映射（当前模式）
     */
    std::pair<cv::Mat, cv::Mat> getRightMaps() const;
    
    /**
     * @brief 获取有效区域ROI
     */
    std::pair<cv::Rect, cv::Rect> getValidROI() const;
    
    /**
     * @brief 获取当前校正模式
     */
    RectificationMode getMode() const { return m_mode; }
    
    /**
     * @brief 设置校正模式
     */
    void setMode(RectificationMode mode);
    
    /**
     * @brief 获取输入图像尺寸
     */
    cv::Size getImageSize() const;
    
    /**
     * @brief 获取输出图像尺寸（校正后的尺寸，通常与输入相同）
     */
    cv::Size getOutputSize() const { return m_output_size; }
    
    /**
     * @brief 检查校正器是否已初始化
     */
    bool isInitialized() const;
    
    /**
     * @brief 重置校正器状态
     */
    void reset();
    
    /**
     * @brief 缩放信息结构体
     */
    struct ScaleInfo {
        cv::Rect roi;           // 原始有效区域
        float scale_factor;     // 缩放因子
        cv::Size scaled_size;   // 缩放后尺寸
        cv::Point offset;       // 填充偏移
        float effective_ratio;  // 有效像素比例
    };
    
    /**
     * @brief 获取缩放信息（仅在SCALE_TO_FIT模式有效）
     */
    ScaleInfo getScaleInfo() const { return m_scale_info; }
    
    /**
     * @brief 保存校正后的图像用于验证（静态工具方法）
     * @param left_rectified 校正后的左眼图像
     * @param right_rectified 校正后的右眼图像
     * @param output_dir 输出目录
     * @param filename 文件名前缀
     * @return 保存成功返回true
     */
    static bool saveRectifiedImages(const cv::Mat& left_rectified,
                                   const cv::Mat& right_rectified,
                                   const std::string& output_dir,
                                   const std::string& filename = "rectified");
    
private:
    /**
     * @brief 计算有效区域ROI
     */
    void computeValidROI();
    
    /**
     * @brief 计算缩放参数（方案2专用）
     */
    void computeScalingParameters();
    
    /**
     * @brief 计算校正映射表（根据当前模式）
     */
    bool computeRectificationMaps();
    
    /**
     * @brief 创建组合映射表（校正+裁剪+缩放+填充）
     */
    bool createCombinedMaps();
    
    /**
     * @brief 更新校正映射表（模式改变时调用）
     */
    bool updateRectificationMaps();
    
    /**
     * @brief 模式转字符串（辅助日志）
     */
    std::string modeToString(RectificationMode mode) const;
    
    // 标定参数
    CalibrationParams m_params;
    
    // 图像尺寸
    cv::Size m_image_size;
    cv::Size m_output_size;
    
    // 原始校正映射表（RAW/CROP_ONLY模式）
    cv::Mat m_left_map1, m_left_map2;
    cv::Mat m_right_map1, m_right_map2;
    
    // 组合映射表（SCALE_TO_FIT模式）
    cv::Mat m_combined_left_map1, m_combined_left_map2;
    cv::Mat m_combined_right_map1, m_combined_right_map2;
    
    // 有效区域
    cv::Rect m_valid_roi_left;
    cv::Rect m_valid_roi_right;
    
    // 缩放信息
    ScaleInfo m_scale_info;
    
    // 当前模式
    RectificationMode m_mode;
    
    // 状态标志
    bool m_initialized;
    bool m_maps_computed;
};

} // namespace calibration
} // namespace stereo_depth
