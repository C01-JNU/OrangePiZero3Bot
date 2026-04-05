#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::Preprocess() : m_gpu(nullptr) {
    LOG_INFO("Preprocess 构造函数 (CPU 模式), m_gpu={}", reinterpret_cast<void*>(m_gpu));
}

Preprocess::~Preprocess() {
    // CPU 模式无需释放 GPU 资源
}

bool Preprocess::initFromConfig() {
    LOG_INFO("Preprocess::initFromConfig 开始 (CPU 模式)");
    if (!m_denoiser.initFromConfig()) {
        LOG_ERROR("去噪器初始化失败");
        return false;
    }

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    CensusParams census_params;
    census_params.window_width = cfg.get<int>("preprocess.census.window_width", 5);
    census_params.window_height = cfg.get<int>("preprocess.census.window_height", 3);
    census_params.adaptive_threshold = cfg.get<int>("preprocess.census.adaptive_threshold", 2);
    census_params.workgroup_size_x = cfg.get<int>("preprocess.gpu.workgroup_size_x", 16);
    census_params.workgroup_size_y = cfg.get<int>("preprocess.gpu.workgroup_size_y", 16);
    if (!m_census.init(census_params)) {
        LOG_ERROR("Census初始化失败");
        return false;
    }

    m_initialized = true;
    LOG_INFO("Preprocess模块初始化完成 (CPU 模式)");
    return true;
}

bool Preprocess::process(const cv::Mat& left, const cv::Mat& right,
                         cv::Mat& left_census, cv::Mat& right_census) {
    if (!m_initialized) {
        LOG_ERROR("Preprocess未初始化");
        return false;
    }
    if (left.empty() || right.empty()) {
        LOG_ERROR("输入图像为空");
        return false;
    }
    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        LOG_ERROR("输入图像必须是CV_8UC3彩色图");
        return false;
    }

    // CPU 模式直接走 processCPU
    return processCPU(left, right, left_census, right_census);
}

bool Preprocess::denoiseOnly(const cv::Mat& src, cv::Mat& dst) {
    return m_denoiser.process(src, dst);
}

bool Preprocess::processCPU(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& left_census, cv::Mat& right_census) {
    LOG_INFO("CPU处理开始");
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    cv::Mat left_denoised, right_denoised;
    if (!m_denoiser.process(left_gray, left_denoised) ||
        !m_denoiser.process(right_gray, right_denoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    if (!m_census.compute(left_denoised, right_denoised, left_census, right_census)) {
        LOG_ERROR("Census计算失败");
        return false;
    }
    LOG_INFO("CPU处理完成");
    return true;
}

bool Preprocess::processGPU(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&) {
    LOG_ERROR("CPU 模式下不应调用 processGPU");
    return false;
}

bool Preprocess::initGPU() {
    LOG_ERROR("CPU 模式下不应调用 initGPU");
    return false;
}

void* Preprocess::getFilteredImageHandle() const {
    return nullptr;
}

int Preprocess::getFilteredImageWidth() const {
    return 0;
}

int Preprocess::getFilteredImageHeight() const {
    return 0;
}

} // namespace stereo_depth::preprocess
