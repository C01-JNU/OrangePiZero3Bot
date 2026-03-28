#include "preprocess/denoiser.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Denoiser::Denoiser() = default;
Denoiser::~Denoiser() = default;

bool Denoiser::initFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    std::string method_str = cfg.get<std::string>("preprocess.denoise.method", "none");
    if (method_str == "median") {
        m_params.method = DenoiseMethod::MEDIAN;
        m_params.median_ksize = cfg.get<int>("preprocess.denoise.median_ksize", 3);
    } else if (method_str == "bilateral") {
        m_params.method = DenoiseMethod::BILATERAL;
        m_params.bilateral_d = cfg.get<int>("preprocess.denoise.bilateral_d", 9);
        m_params.bilateral_sigma_color = cfg.get<double>("preprocess.denoise.bilateral_sigma_color", 50.0);
        m_params.bilateral_sigma_space = cfg.get<double>("preprocess.denoise.bilateral_sigma_space", 9.0);
    } else if (method_str == "tinylut") {
        m_params.method = DenoiseMethod::TINYLUT;
        m_params.tinylut_table_dir = cfg.get<std::string>("preprocess.denoise.tinylut_table_dir", "model/tinylut");
        LOG_WARN("TinyLUT denoiser not implemented yet, will return original image");
    } else {
        m_params.method = DenoiseMethod::NONE;
    }
    LOG_INFO("Denoiser initialized: method={}", method_str);
    return true;
}

bool Denoiser::setParams(const DenoiseParams& params) {
    m_params = params;
    return true;
}

bool Denoiser::process(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        LOG_ERROR("Denoiser: input empty");
        return false;
    }
    if (src.type() != CV_8UC1) {
        LOG_ERROR("Denoiser: input must be CV_8UC1");
        return false;
    }

    switch (m_params.method) {
        case DenoiseMethod::MEDIAN:
            cv::medianBlur(src, dst, m_params.median_ksize);
            break;
        case DenoiseMethod::BILATERAL:
            cv::bilateralFilter(src, dst, m_params.bilateral_d,
                                m_params.bilateral_sigma_color,
                                m_params.bilateral_sigma_space);
            break;
        case DenoiseMethod::TINYLUT:
            // 预留：加载 .npy 表并查表
            LOG_WARN("TinyLUT not implemented, returning original image");
            dst = src.clone();
            break;
        default:
            dst = src.clone();
            break;
    }
    return true;
}

} // namespace stereo_depth::preprocess
