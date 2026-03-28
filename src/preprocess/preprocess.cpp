#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"

namespace stereo_depth::preprocess {

Preprocess::Preprocess() = default;
Preprocess::~Preprocess() = default;

bool Preprocess::initFromConfig() {
    if (!m_denoiser.initFromConfig()) {
        LOG_ERROR("Failed to init denoiser");
        return false;
    }

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    CensusParams census_params;
    census_params.window_width = cfg.get<int>("preprocess.census.window_width", 5);
    census_params.window_height = cfg.get<int>("preprocess.census.window_height", 3);
    census_params.adaptive_threshold = cfg.get<int>("preprocess.census.adaptive_threshold", 2);
    if (!m_census.init(census_params)) {
        LOG_ERROR("Failed to init Census");
        return false;
    }

    m_initialized = true;
    LOG_INFO("Preprocess module initialized");
    return true;
}

bool Preprocess::process(const cv::Mat& left, const cv::Mat& right,
                         cv::Mat& left_census, cv::Mat& right_census) {
    if (!m_initialized) {
        LOG_ERROR("Preprocess not initialized");
        return false;
    }
    if (left.empty() || right.empty()) {
        LOG_ERROR("Input images empty");
        return false;
    }
    if (left.type() != CV_8UC1 || right.type() != CV_8UC1) {
        LOG_ERROR("Input must be CV_8UC1");
        return false;
    }

    cv::Mat left_denoised, right_denoised;
    if (!m_denoiser.process(left, left_denoised) ||
        !m_denoiser.process(right, right_denoised)) {
        LOG_ERROR("Denoise failed");
        return false;
    }

    if (!m_census.compute(left_denoised, right_denoised, left_census, right_census)) {
        LOG_ERROR("Census failed");
        return false;
    }
    return true;
}

bool Preprocess::denoiseOnly(const cv::Mat& src, cv::Mat& dst) {
    return m_denoiser.process(src, dst);
}

} // namespace stereo_depth::preprocess
