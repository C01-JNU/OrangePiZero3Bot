#include "cpu_stereo/cpu_stereo_matcher.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace stereo_depth {
namespace cpu_stereo {

CpuStereoMatcher::CpuStereoMatcher() = default;

CpuStereoMatcher::~CpuStereoMatcher() = default;

bool CpuStereoMatcher::initializeFromConfig(const std::string& config_path) {
    auto& cfg_mgr = stereo_depth::utils::ConfigManager::getInstance();
    if (!config_path.empty()) {
        if (!cfg_mgr.loadGlobalConfig(config_path)) {
            LOG_ERROR("CpuStereoMatcher: 无法加载配置文件: {}", config_path);
            return false;
        }
    }
    const auto& cfg = cfg_mgr.getConfig();

    algorithm_ = cfg.get<std::string>("stereo.algorithm", "sgbm");
    LOG_INFO("CPU立体匹配器初始化，算法: {}", algorithm_);

    disparity_range_ = cfg.get<int>("stereo.disparity_range", 64);
    min_disparity_ = cfg.get<int>("stereo.min_disparity", 0);
    median_filter_size_ = cfg.get<int>("stereo.median_filter_size", 3);
    if (median_filter_size_ % 2 == 0) median_filter_size_++;

    // 读取通用后处理参数（用于SGBM和BM）
    uniqueness_ratio_ = cfg.get<int>("stereo.uniqueness_ratio", 15);
    speckle_window_size_ = cfg.get<int>("stereo.speckle_window_size", 0);
    speckle_range_ = cfg.get<int>("stereo.speckle_range", 0);
    disp12_max_diff_ = cfg.get<int>("stereo.disp12_max_diff", -1);
    pre_filter_cap_ = cfg.get<int>("stereo.pre_filter_cap", 0);

    if (algorithm_ == "sgbm") {
        sgbm_params_.block_size = cfg.get<int>("stereo.sgbm.block_size", 7);
        if (sgbm_params_.block_size % 2 == 0) sgbm_params_.block_size++;
        sgbm_params_.p1 = cfg.get<double>("stereo.sgbm.p1", 8.0);
        sgbm_params_.p2 = cfg.get<double>("stereo.sgbm.p2", 32.0);
        sgbm_params_.mode = cfg.get<bool>("stereo.sgbm.mode", true);
    } else if (algorithm_ == "bm") {
        bm_params_.block_size = cfg.get<int>("stereo.bm.block_size", 7);
        if (bm_params_.block_size % 2 == 0) bm_params_.block_size++;
        bm_params_.pre_filter_type = cfg.get<int>("stereo.bm.pre_filter_type", 1);
        bm_params_.pre_filter_size = cfg.get<int>("stereo.bm.pre_filter_size", 9);
        bm_params_.pre_filter_cap = cfg.get<int>("stereo.bm.pre_filter_cap", 31);
        bm_params_.texture_threshold = cfg.get<int>("stereo.bm.texture_threshold", 10);
        bm_params_.uniqueness_ratio = cfg.get<int>("stereo.bm.uniqueness_ratio", 15);
        bm_params_.speckle_window_size = cfg.get<int>("stereo.bm.speckle_window_size", 100);
        bm_params_.speckle_range = cfg.get<int>("stereo.bm.speckle_range", 32);
        bm_params_.try_small_disp = cfg.get<bool>("stereo.bm.try_small_disp", false);
    } else {
        LOG_ERROR("不支持的算法: {}, 使用sgbm作为默认", algorithm_);
        algorithm_ = "sgbm";
    }

    if (!createMatcher()) {
        LOG_ERROR("创建匹配器失败");
        return false;
    }

    LOG_INFO("CPU立体匹配器初始化完成");
    return true;
}

bool CpuStereoMatcher::createMatcher() {
    try {
        if (algorithm_ == "sgbm") {
            int mode = sgbm_params_.mode ? cv::StereoSGBM::MODE_SGBM_3WAY : 0;
            auto sgbm = cv::StereoSGBM::create(
                min_disparity_, disparity_range_,
                sgbm_params_.block_size,
                static_cast<int>(sgbm_params_.p1),
                static_cast<int>(sgbm_params_.p2),
                disp12_max_diff_,
                pre_filter_cap_,
                uniqueness_ratio_,
                speckle_window_size_,
                speckle_range_,
                mode
            );
            matcher_ = sgbm;
            LOG_INFO("创建StereoSGBM匹配器: blockSize={}, P1={}, P2={}, uniquenessRatio={}, speckleWindow={}, speckleRange={}",
                     sgbm_params_.block_size, sgbm_params_.p1, sgbm_params_.p2,
                     uniqueness_ratio_, speckle_window_size_, speckle_range_);
        } else if (algorithm_ == "bm") {
            auto bm = cv::StereoBM::create(disparity_range_, bm_params_.block_size);
            bm->setPreFilterType(bm_params_.pre_filter_type);
            bm->setPreFilterSize(bm_params_.pre_filter_size);
            bm->setPreFilterCap(bm_params_.pre_filter_cap);
            bm->setTextureThreshold(bm_params_.texture_threshold);
            bm->setUniquenessRatio(bm_params_.uniqueness_ratio);
            bm->setSpeckleWindowSize(bm_params_.speckle_window_size);
            bm->setSpeckleRange(bm_params_.speckle_range);
            bm->setDisp12MaxDiff(disp12_max_diff_);
            bm->setMinDisparity(min_disparity_);
            matcher_ = bm;
            LOG_INFO("创建StereoBM匹配器: blockSize={}, uniquenessRatio={}, speckleWindow={}, speckleRange={}",
                     bm_params_.block_size, bm_params_.uniqueness_ratio,
                     bm_params_.speckle_window_size, bm_params_.speckle_range);
        } else {
            LOG_ERROR("未知算法");
            return false;
        }
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV异常: {}", e.what());
        return false;
    }
    return true;
}

cv::Mat CpuStereoMatcher::compute(const cv::Mat& left, const cv::Mat& right) {
    if (!matcher_) {
        LOG_ERROR("匹配器未初始化");
        return cv::Mat();
    }

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat disp16;
    matcher_->compute(left, right, disp16);

    if (median_filter_size_ >= 3) {
        cv::medianBlur(disp16, disp16, median_filter_size_);
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    return disp16;
}

} // namespace cpu_stereo
} // namespace stereo_depth
