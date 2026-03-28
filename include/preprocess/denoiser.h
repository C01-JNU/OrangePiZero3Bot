// denoiser.h
// 去噪器接口，支持中值滤波、双边滤波、TinyLUT 查表（预留）
// 最后更新: 2026-03-28

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <memory>

namespace stereo_depth::preprocess {

enum class DenoiseMethod {
    NONE,
    MEDIAN,
    BILATERAL,
    TINYLUT   // 预留
};

struct DenoiseParams {
    DenoiseMethod method = DenoiseMethod::NONE;

    // 中值滤波参数
    int median_ksize = 3;

    // 双边滤波参数
    int bilateral_d = 9;
    double bilateral_sigma_color = 50.0;
    double bilateral_sigma_space = 9.0;

    // TinyLUT 参数（预留）
    std::string tinylut_table_dir;      // 存放 .npy 表的目录
};

class Denoiser {
public:
    Denoiser();
    ~Denoiser();

    // 从配置文件初始化
    bool initFromConfig();

    // 设置参数（也可直接调用）
    bool setParams(const DenoiseParams& params);
    // 对单张灰度图去噪
    bool process(const cv::Mat& src, cv::Mat& dst);

private:
    DenoiseParams m_params;
    // TinyLUT 资源（预留），空结构体以保证 unique_ptr 可析构
    struct TinyLutResources {};
    std::unique_ptr<TinyLutResources> m_tinylut;
};

} // namespace stereo_depth::preprocess
