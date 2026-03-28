#include "calibration/stereo_rectifier.hpp"
#include "utils/logger.hpp"
#include <sys/stat.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <errno.h>
#include <fstream>

namespace stereo_depth {
namespace calibration {

// ==================== 构造函数/析构函数 ====================

StereoRectifier::StereoRectifier(RectificationMode mode)
    : m_mode(mode)
    , m_has_calib_roi(false)
    , m_initialized(false)
    , m_maps_computed(false) {
    LOG_DEBUG("StereoRectifier 创建，默认模式: {}", modeToString(m_mode));
}

StereoRectifier::StereoRectifier(const std::string& calibration_file,
                                 RectificationMode mode)
    : m_mode(mode)
    , m_has_calib_roi(false)
    , m_initialized(false)
    , m_maps_computed(false) {
    loadAndInitialize(calibration_file, mode);
}

StereoRectifier::~StereoRectifier() = default;

// ==================== 公共初始化接口 ====================

bool StereoRectifier::initialize(const CalibrationParams& params,
                                 RectificationMode mode) {
    if (!params.isValid()) {
        LOG_ERROR("无效的标定参数，无法初始化校正器");
        return false;
    }

    m_params = params;
    m_image_size = params.image_size;
    m_mode = mode;
    m_output_size = m_image_size;

    LOG_INFO("初始化立体校正处理器 (模式: {})", modeToString(m_mode));
    LOG_INFO("  输入尺寸: {}x{}", m_image_size.width, m_image_size.height);
    LOG_INFO("  基线长度: {:.2f} mm", m_params.baseline_meters * 1000.0);

    if (!m_has_calib_roi) {
        LOG_INFO("未从标定文件加载ROI，使用后备计算方法");
        computeValidROI();
    } else {
        LOG_INFO("使用从标定文件加载的ROI，跳过后备计算");
    }

    if (m_mode == RectificationMode::SCALE_TO_FIT) {
        computeScalingParameters();
    }

    if (!computeRectificationMaps()) {
        LOG_ERROR("计算校正映射表失败");
        return false;
    }

    m_initialized = true;
    LOG_INFO("立体校正处理器初始化完成");
    return true;
}

bool StereoRectifier::loadAndInitialize(const std::string& calibration_file,
                                        RectificationMode mode) {
    CalibrationParams params;
    CalibrationLoader loader;
    if (!loader.loadFromFile(calibration_file, params)) {
        LOG_ERROR("加载标定文件失败: {}", calibration_file);
        return false;
    }

    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (fs.isOpened()) {
        cv::Rect roi_left, roi_right;
        fs["valid_roi_left"] >> roi_left;
        fs["valid_roi_right"] >> roi_right;

        LOG_INFO("从标定文件读取的 ROI 左: ({},{},{},{}), 右: ({},{},{},{})",
                 roi_left.x, roi_left.y, roi_left.width, roi_left.height,
                 roi_right.x, roi_right.y, roi_right.width, roi_right.height);

        if (!roi_left.empty() && !roi_right.empty()) {
            cv::Rect common_roi = roi_left & roi_right;
            if (common_roi.area() > 0) {
                m_valid_roi_left = common_roi;
                m_valid_roi_right = common_roi;
                m_has_calib_roi = true;
                LOG_INFO("从标定文件加载有效区域 ROI，取公共区域后: ({},{}) {}x{}",
                         common_roi.x, common_roi.y, common_roi.width, common_roi.height);
            } else {
                LOG_WARN("标定文件中左右 ROI 无交集，将使用计算出的ROI");
            }
        } else {
            LOG_WARN("标定文件中未找到 valid_roi_left/right，将使用计算出的ROI");
        }
        fs.release();
    } else {
        LOG_WARN("无法重新打开标定文件读取ROI，将使用计算出的ROI");
    }

    return initialize(params, mode);
}

// ==================== 校正主接口 ====================

bool StereoRectifier::rectifyPair(const cv::Mat& left_image,
                                   const cv::Mat& right_image,
                                   cv::Mat& left_rectified,
                                   cv::Mat& right_rectified) {
    if (!m_initialized || !m_maps_computed) {
        LOG_ERROR("校正器未初始化或映射表未计算");
        return false;
    }

    if (left_image.empty() || right_image.empty()) {
        LOG_ERROR("输入图像为空");
        return false;
    }

    if (left_image.size() != m_image_size || right_image.size() != m_image_size) {
        LOG_ERROR("输入图像尺寸不匹配: 期望 {}x{}, 实际左={}x{}, 右={}x{}",
                  m_image_size.width, m_image_size.height,
                  left_image.cols, left_image.rows,
                  right_image.cols, right_image.rows);
        return false;
    }

    // 移除单通道限制，允许任意通道（灰度或彩色）
    // remap 会自动处理多通道图像

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        if (m_mode == RectificationMode::SCALE_TO_FIT && !m_combined_left_map1.empty()) {
            cv::remap(left_image, left_rectified,
                      m_combined_left_map1, m_combined_left_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            cv::remap(right_image, right_rectified,
                      m_combined_right_map1, m_combined_right_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        } else if (m_mode == RectificationMode::CROP_ONLY) {
            cv::Mat left_tmp, right_tmp;
            cv::remap(left_image, left_tmp,
                      m_left_map1, m_left_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            cv::remap(right_image, right_tmp,
                      m_right_map1, m_right_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

            left_rectified = left_tmp(m_valid_roi_left).clone();
            right_rectified = right_tmp(m_valid_roi_right).clone();

            m_output_size = left_rectified.size();
        } else { // RAW
            cv::remap(left_image, left_rectified,
                      m_left_map1, m_left_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            cv::remap(right_image, right_rectified,
                      m_right_map1, m_right_map2,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            m_output_size = m_image_size;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        LOG_DEBUG("立体校正完成: 模式={}, 耗时={:.2f} ms",
                  modeToString(m_mode), duration.count() / 1000.0);
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV校正异常: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("标准异常: {}", e.what());
        return false;
    }
}

bool StereoRectifier::rectifyBatch(
        const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
        std::vector<std::pair<cv::Mat, cv::Mat>>& rectified_pairs) {

    if (!m_initialized) {
        LOG_ERROR("校正器未初始化");
        return false;
    }

    if (image_pairs.empty()) {
        LOG_WARN("输入图像对列表为空");
        rectified_pairs.clear();
        return true;
    }

    LOG_INFO("批量校正 {} 对图像", image_pairs.size());

    auto total_start = std::chrono::high_resolution_clock::now();
    rectified_pairs.clear();
    rectified_pairs.reserve(image_pairs.size());

    bool all_success = true;
    size_t success_count = 0;

    for (size_t i = 0; i < image_pairs.size(); ++i) {
        const auto& [left, right] = image_pairs[i];

        cv::Mat left_rect, right_rect;
        if (rectifyPair(left, right, left_rect, right_rect)) {
            rectified_pairs.emplace_back(left_rect, right_rect);
            success_count++;

            if ((i + 1) % 10 == 0) {
                LOG_DEBUG("已处理 {}/{} 对图像", i + 1, image_pairs.size());
            }
        } else {
            LOG_ERROR("第 {} 对图像校正失败", i + 1);
            all_success = false;
            rectified_pairs.emplace_back(cv::Mat(), cv::Mat());
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    double avg_time = success_count > 0 ? total_duration.count() / static_cast<double>(success_count) : 0.0;
    LOG_INFO("批量校正完成: {}/{} 成功, 总耗时: {:.2f} ms, 平均: {:.2f} ms/对",
             success_count, image_pairs.size(), total_duration.count(), avg_time);

    return all_success;
}

// ==================== 校正参数计算 ====================

void StereoRectifier::computeValidROI() {
    LOG_INFO("计算有效区域ROI (后备方法)...");

    cv::Mat test_image = cv::Mat::zeros(m_image_size, CV_8UC1);

    cv::Mat left_map1, left_map2, right_map1, right_map2;
    cv::initUndistortRectifyMap(
        m_params.camera_matrix_left,
        m_params.dist_coeffs_left,
        m_params.rectification_left,
        m_params.projection_left,
        m_image_size,
        CV_32FC1,
        left_map1,
        left_map2
    );

    cv::initUndistortRectifyMap(
        m_params.camera_matrix_right,
        m_params.dist_coeffs_right,
        m_params.rectification_right,
        m_params.projection_right,
        m_image_size,
        CV_32FC1,
        right_map1,
        right_map2
    );

    cv::Mat rectified_left, rectified_right;
    cv::remap(test_image, rectified_left, left_map1, left_map2, cv::INTER_LINEAR);
    cv::remap(test_image, rectified_right, right_map1, right_map2, cv::INTER_LINEAR);

    std::vector<cv::Point> left_points, right_points;
    cv::findNonZero(rectified_left > 0, left_points);
    cv::findNonZero(rectified_right > 0, right_points);

    if (left_points.empty() || right_points.empty()) {
        LOG_WARN("校正后图像完全无效（可能由于投影矩阵问题），使用全图作为ROI");
        m_valid_roi_left = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        m_valid_roi_right = cv::Rect(0, 0, m_image_size.width, m_image_size.height);
        return;
    }

    cv::Rect roi_left = cv::boundingRect(left_points);
    cv::Rect roi_right = cv::boundingRect(right_points);

    cv::Rect common_roi = roi_left & roi_right;
    if (common_roi.area() > 0) {
        m_valid_roi_left = common_roi;
        m_valid_roi_right = common_roi;
    } else {
        m_valid_roi_left = roi_left;
        m_valid_roi_right = roi_right;
    }

    LOG_INFO("计算出的有效区域ROI:");
    LOG_INFO("  左眼: x={}, y={}, {}x{} (原图{:.1f}%)",
             m_valid_roi_left.x, m_valid_roi_left.y,
             m_valid_roi_left.width, m_valid_roi_left.height,
             (m_valid_roi_left.area() * 100.0) / (m_image_size.area()));
    LOG_INFO("  右眼: x={}, y={}, {}x{} (原图{:.1f}%)",
             m_valid_roi_right.x, m_valid_roi_right.y,
             m_valid_roi_right.width, m_valid_roi_right.height,
             (m_valid_roi_right.area() * 100.0) / (m_image_size.area()));
}

void StereoRectifier::computeScalingParameters() {
    LOG_INFO("计算缩放参数 (SCALE_TO_FIT模式)...");

    cv::Rect roi = m_valid_roi_left;
    m_scale_info.roi = roi;

    float scale_x = static_cast<float>(m_image_size.width) / roi.width;
    float scale_y = static_cast<float>(m_image_size.height) / roi.height;
    m_scale_info.scale_factor = std::min(scale_x, scale_y);

    m_scale_info.scaled_size.width = static_cast<int>(roi.width * m_scale_info.scale_factor + 0.5f);
    m_scale_info.scaled_size.height = static_cast<int>(roi.height * m_scale_info.scale_factor + 0.5f);

    m_scale_info.offset.x = (m_image_size.width - m_scale_info.scaled_size.width) / 2;
    m_scale_info.offset.y = (m_image_size.height - m_scale_info.scaled_size.height) / 2;

    float original_effective = roi.area() / static_cast<float>(m_image_size.area());
    float scaled_effective = m_scale_info.scaled_size.area() / static_cast<float>(m_image_size.area());
    m_scale_info.effective_ratio = scaled_effective;

    LOG_INFO("缩放参数:");
    LOG_INFO("  原始ROI: {}x{} (位置: {}, {})", roi.width, roi.height, roi.x, roi.y);
    LOG_INFO("  缩放因子: {:.3f}", m_scale_info.scale_factor);
    LOG_INFO("  缩放后尺寸: {}x{}", m_scale_info.scaled_size.width, m_scale_info.scaled_size.height);
    LOG_INFO("  填充偏移: ({}, {})", m_scale_info.offset.x, m_scale_info.offset.y);
    LOG_INFO("  有效像素比例: 原始{:.1f}% → 缩放后{:.1f}%",
             original_effective * 100.0, scaled_effective * 100.0);
}

bool StereoRectifier::computeRectificationMaps() {
    LOG_INFO("计算校正映射表 (模式: {})...", modeToString(m_mode));

    try {
        if (m_mode == RectificationMode::SCALE_TO_FIT) {
            return createCombinedMaps();
        } else {
            cv::initUndistortRectifyMap(
                m_params.camera_matrix_left,
                m_params.dist_coeffs_left,
                m_params.rectification_left,
                m_params.projection_left,
                m_image_size,
                CV_32FC1,
                m_left_map1,
                m_left_map2
            );

            cv::initUndistortRectifyMap(
                m_params.camera_matrix_right,
                m_params.dist_coeffs_right,
                m_params.rectification_right,
                m_params.projection_right,
                m_image_size,
                CV_32FC1,
                m_right_map1,
                m_right_map2
            );

            LOG_INFO("原始校正映射表计算完成");
            m_output_size = m_image_size;
            m_maps_computed = true;
            return true;
        }
    } catch (const cv::Exception& e) {
        LOG_ERROR("计算校正映射表异常: {}", e.what());
        return false;
    }
}

bool StereoRectifier::createCombinedMaps() {
    LOG_INFO("创建组合映射表（校正+裁剪+缩放+填充）...");

    try {
        cv::Mat left_map1, left_map2, right_map1, right_map2;
        cv::initUndistortRectifyMap(
            m_params.camera_matrix_left,
            m_params.dist_coeffs_left,
            m_params.rectification_left,
            m_params.projection_left,
            m_image_size,
            CV_32FC1,
            left_map1,
            left_map2
        );

        cv::initUndistortRectifyMap(
            m_params.camera_matrix_right,
            m_params.dist_coeffs_right,
            m_params.rectification_right,
            m_params.projection_right,
            m_image_size,
            CV_32FC1,
            right_map1,
            right_map2
        );

        m_combined_left_map1 = cv::Mat(m_image_size, CV_32FC1, cv::Scalar(0));
        m_combined_left_map2 = cv::Mat(m_image_size, CV_32FC1, cv::Scalar(0));
        m_combined_right_map1 = cv::Mat(m_image_size, CV_32FC1, cv::Scalar(0));
        m_combined_right_map2 = cv::Mat(m_image_size, CV_32FC1, cv::Scalar(0));

        for (int y = 0; y < m_image_size.height; ++y) {
            for (int x = 0; x < m_image_size.width; ++x) {
                float scaled_x = (static_cast<float>(x - m_scale_info.offset.x)) / m_scale_info.scale_factor;
                float scaled_y = (static_cast<float>(y - m_scale_info.offset.y)) / m_scale_info.scale_factor;

                float rectified_x = scaled_x + m_scale_info.roi.x;
                float rectified_y = scaled_y + m_scale_info.roi.y;

                if (rectified_x >= 0 && rectified_x < m_image_size.width &&
                    rectified_y >= 0 && rectified_y < m_image_size.height) {
                    int rx = static_cast<int>(rectified_x + 0.5f);
                    int ry = static_cast<int>(rectified_y + 0.5f);

                    rx = std::min(std::max(rx, 0), m_image_size.width - 1);
                    ry = std::min(std::max(ry, 0), m_image_size.height - 1);

                    m_combined_left_map1.at<float>(y, x) = left_map1.at<float>(ry, rx);
                    m_combined_left_map2.at<float>(y, x) = left_map2.at<float>(ry, rx);
                    m_combined_right_map1.at<float>(y, x) = right_map1.at<float>(ry, rx);
                    m_combined_right_map2.at<float>(y, x) = right_map2.at<float>(ry, rx);
                } else {
                    m_combined_left_map1.at<float>(y, x) = -1.0f;
                    m_combined_left_map2.at<float>(y, x) = -1.0f;
                    m_combined_right_map1.at<float>(y, x) = -1.0f;
                    m_combined_right_map2.at<float>(y, x) = -1.0f;
                }
            }
        }

        m_output_size = m_image_size;
        m_maps_computed = true;

        LOG_INFO("组合映射表创建完成");
        LOG_DEBUG("有效映射点比例: {:.1f}%", 100.0f * m_scale_info.effective_ratio);

        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("创建组合映射表异常: {}", e.what());
        return false;
    }
}

bool StereoRectifier::updateRectificationMaps() {
    LOG_INFO("更新校正映射表 (模式改变为 {})", modeToString(m_mode));

    if (m_mode == RectificationMode::SCALE_TO_FIT) {
        computeScalingParameters();
    }

    m_maps_computed = false;
    return computeRectificationMaps();
}

// ==================== 辅助方法 ====================

void StereoRectifier::setMode(RectificationMode mode) {
    if (mode != m_mode) {
        LOG_INFO("切换校正模式: {} -> {}", modeToString(m_mode), modeToString(mode));
        m_mode = mode;
        if (m_initialized) {
            updateRectificationMaps();
        }
    }
}

std::string StereoRectifier::modeToString(RectificationMode mode) const {
    switch (mode) {
        case RectificationMode::RAW:         return "RAW";
        case RectificationMode::CROP_ONLY:   return "CROP_ONLY";
        case RectificationMode::SCALE_TO_FIT: return "SCALE_TO_FIT";
        default:                             return "UNKNOWN";
    }
}

void StereoRectifier::reset() {
    LOG_INFO("重置立体校正处理器");

    m_params = CalibrationParams();
    m_image_size = cv::Size(0, 0);
    m_output_size = cv::Size(0, 0);

    m_left_map1.release();
    m_left_map2.release();
    m_right_map1.release();
    m_right_map2.release();

    m_combined_left_map1.release();
    m_combined_left_map2.release();
    m_combined_right_map1.release();
    m_combined_right_map2.release();

    m_valid_roi_left = cv::Rect();
    m_valid_roi_right = cv::Rect();
    m_has_calib_roi = false;

    m_scale_info = ScaleInfo();

    m_initialized = false;
    m_maps_computed = false;
}

// ==================== 静态工具方法 ====================

bool StereoRectifier::saveRectifiedImages(const cv::Mat& left_rectified,
                                           const cv::Mat& right_rectified,
                                           const std::string& output_dir,
                                           const std::string& filename) {
    struct stat st;
    if (stat(output_dir.c_str(), &st) != 0) {
        if (mkdir(output_dir.c_str(), 0777) != 0 && errno != EEXIST) {
            LOG_ERROR("无法创建输出目录: {}", output_dir);
            return false;
        }
    }

    try {
        std::string left_path = output_dir + "/" + filename + "_left.png";
        std::string right_path = output_dir + "/" + filename + "_right.png";

        if (!cv::imwrite(left_path, left_rectified)) {
            LOG_ERROR("无法保存左校正图像: {}", left_path);
            return false;
        }

        if (!cv::imwrite(right_path, right_rectified)) {
            LOG_ERROR("无法保存右校正图像: {}", right_path);
            return false;
        }

        LOG_DEBUG("校正图像已保存: {} 和 {}", left_path, right_path);
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("保存校正图像异常: {}", e.what());
        return false;
    }
}

// ==================== 获取器实现 ====================

const CalibrationParams& StereoRectifier::getCalibrationParams() const {
    return m_params;
}

std::pair<cv::Mat, cv::Mat> StereoRectifier::getLeftMaps() const {
    if (m_mode == RectificationMode::SCALE_TO_FIT && !m_combined_left_map1.empty()) {
        return {m_combined_left_map1, m_combined_left_map2};
    } else {
        return {m_left_map1, m_left_map2};
    }
}

std::pair<cv::Mat, cv::Mat> StereoRectifier::getRightMaps() const {
    if (m_mode == RectificationMode::SCALE_TO_FIT && !m_combined_right_map1.empty()) {
        return {m_combined_right_map1, m_combined_right_map2};
    } else {
        return {m_right_map1, m_right_map2};
    }
}

std::pair<cv::Rect, cv::Rect> StereoRectifier::getValidROI() const {
    return {m_valid_roi_left, m_valid_roi_right};
}

cv::Size StereoRectifier::getImageSize() const {
    return m_image_size;
}

cv::Size StereoRectifier::getOutputSize() const {
    if (m_mode == RectificationMode::CROP_ONLY && m_initialized) {
        return m_valid_roi_left.size();
    }
    return m_output_size;
}

bool StereoRectifier::isInitialized() const {
    return m_initialized;
}

} // namespace calibration
} // namespace stereo_depth
