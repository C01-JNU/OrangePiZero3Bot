#pragma once

#include "calibration/calibration_loader.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <utility>

namespace stereo_depth {
namespace calibration {

enum class RectificationMode {
    RAW,
    CROP_ONLY,
    SCALE_TO_FIT
};

class StereoRectifier {
public:
    StereoRectifier(RectificationMode mode = RectificationMode::CROP_ONLY);
    explicit StereoRectifier(const std::string& calibration_file,
                             RectificationMode mode = RectificationMode::CROP_ONLY);
    ~StereoRectifier();

    StereoRectifier(const StereoRectifier&) = delete;
    StereoRectifier& operator=(const StereoRectifier&) = delete;
    StereoRectifier(StereoRectifier&&) = default;
    StereoRectifier& operator=(StereoRectifier&&) = default;

    bool initialize(const CalibrationParams& params,
                    RectificationMode mode = RectificationMode::CROP_ONLY);
    bool loadAndInitialize(const std::string& calibration_file,
                           RectificationMode mode = RectificationMode::CROP_ONLY);

    /**
     * @brief 对双目图像对进行立体校正
     * @param left_image  左目图像（支持 CV_8UC1 或 CV_8UC3）
     * @param right_image 右目图像（支持 CV_8UC1 或 CV_8UC3）
     * @param left_rectified  输出校正后的左图（与输入类型相同）
     * @param right_rectified 输出校正后的右图
     * @return 成功返回 true
     */
    bool rectifyPair(const cv::Mat& left_image,
                     const cv::Mat& right_image,
                     cv::Mat& left_rectified,
                     cv::Mat& right_rectified);

    bool rectifyBatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                      std::vector<std::pair<cv::Mat, cv::Mat>>& rectified_pairs);

    const CalibrationParams& getCalibrationParams() const;
    std::pair<cv::Mat, cv::Mat> getLeftMaps() const;
    std::pair<cv::Mat, cv::Mat> getRightMaps() const;
    std::pair<cv::Rect, cv::Rect> getValidROI() const;
    RectificationMode getMode() const { return m_mode; }
    void setMode(RectificationMode mode);
    cv::Size getImageSize() const;
    cv::Size getOutputSize() const;
    bool isInitialized() const;
    void reset();

    struct ScaleInfo {
        cv::Rect roi;
        float scale_factor;
        cv::Size scaled_size;
        cv::Point offset;
        float effective_ratio;
    };
    ScaleInfo getScaleInfo() const { return m_scale_info; }

    static bool saveRectifiedImages(const cv::Mat& left_rectified,
                                    const cv::Mat& right_rectified,
                                    const std::string& output_dir,
                                    const std::string& filename = "rectified");

private:
    void computeValidROI();
    void computeScalingParameters();
    bool computeRectificationMaps();
    bool createCombinedMaps();
    bool updateRectificationMaps();
    std::string modeToString(RectificationMode mode) const;

    CalibrationParams m_params;
    cv::Size m_image_size;
    cv::Size m_output_size;

    cv::Mat m_left_map1, m_left_map2;
    cv::Mat m_right_map1, m_right_map2;

    cv::Mat m_combined_left_map1, m_combined_left_map2;
    cv::Mat m_combined_right_map1, m_combined_right_map2;

    cv::Rect m_valid_roi_left;
    cv::Rect m_valid_roi_right;
    bool m_has_calib_roi;

    ScaleInfo m_scale_info;

    RectificationMode m_mode;
    bool m_initialized;
    bool m_maps_computed;
};

} // namespace calibration
} // namespace stereo_depth
