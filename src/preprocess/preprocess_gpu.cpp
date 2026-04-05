#include "preprocess/preprocess.h"
#include "preprocess/gpu_census.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::preprocess {

Preprocess::~Preprocess() {
    delete m_censusGpu;
}

bool Preprocess::initGpu() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    // 必须从配置读取，不允许硬编码默认值，如果缺失则报错
    if (!cfg.has("preprocess.census.window_width")) {
        LOG_ERROR("配置缺失: preprocess.census.window_width");
        return false;
    }
    if (!cfg.has("preprocess.census.window_height")) {
        LOG_ERROR("配置缺失: preprocess.census.window_height");
        return false;
    }
    if (!cfg.has("preprocess.census.adaptive_threshold")) {
        LOG_ERROR("配置缺失: preprocess.census.adaptive_threshold");
        return false;
    }
    int windowWidth = cfg.get<int>("preprocess.census.window_width");
    int windowHeight = cfg.get<int>("preprocess.census.window_height");
    int adaptiveThreshold = cfg.get<int>("preprocess.census.adaptive_threshold");

    // 验证参数有效性
    if (windowWidth <= 0 || windowWidth % 2 == 0 || windowHeight <= 0 || windowHeight % 2 == 0) {
        LOG_ERROR("Census 窗口尺寸必须为正奇数");
        return false;
    }

    m_censusGpu = new GpuCensusTransform();
    if (!m_censusGpu->init(windowWidth, windowHeight, adaptiveThreshold)) {
        LOG_ERROR("GPU Census 初始化失败");
        delete m_censusGpu;
        m_censusGpu = nullptr;
        return false;
    }
    return true;
}

bool Preprocess::processGpu(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& leftCensus, cv::Mat& rightCensus) {
    if (!m_censusGpu) {
        LOG_ERROR("GPU Census 未初始化");
        return false;
    }

    // 先 CPU 去噪（彩色图转灰度去噪）
    cv::Mat leftGray, rightGray;
    cv::cvtColor(left, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, rightGray, cv::COLOR_BGR2GRAY);

    cv::Mat leftDenoised, rightDenoised;
    if (!m_denoiser.process(leftGray, leftDenoised) ||
        !m_denoiser.process(rightGray, rightDenoised)) {
        LOG_ERROR("去噪失败");
        return false;
    }

    // 去噪后的灰度图转换为彩色图（三通道相同）供 GPU Census 使用
    cv::Mat leftDenoisedColor, rightDenoisedColor;
    cv::cvtColor(leftDenoised, leftDenoisedColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rightDenoised, rightDenoisedColor, cv::COLOR_GRAY2BGR);

    // GPU Census
    if (!m_censusGpu->process(leftDenoisedColor, leftCensus) ||
        !m_censusGpu->process(rightDenoisedColor, rightCensus)) {
        LOG_ERROR("Census GPU 失败");
        return false;
    }
    return true;
}

void* Preprocess::getFilteredImageHandle() const {
    if (m_censusGpu) {
        return m_censusGpu->getOutputImageView();
    }
    return nullptr;
}
int Preprocess::getFilteredImageWidth() const {
    if (m_censusGpu) {
        return m_censusGpu->getWidth();
    }
    return 0;
}
int Preprocess::getFilteredImageHeight() const {
    if (m_censusGpu) {
        return m_censusGpu->getHeight();
    }
    return 0;
}

} // namespace stereo_depth::preprocess
