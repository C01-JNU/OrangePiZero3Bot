#include "census/filters.h"
#include "utils/logger.hpp"
#include <opencv2/imgproc.hpp>

namespace stereo_depth::census {

bool applyFilterCPU(const cv::Mat& src, cv::Mat& dst, const FilterParams& params) {
    if (src.empty()) {
        LOG_ERROR("applyFilterCPU: input image is empty");
        return false;
    }
    if (src.type() != CV_8UC1) {
        LOG_ERROR("applyFilterCPU: input must be CV_8UC1");
        return false;
    }

    switch (params.type) {
        case FilterType::MEDIAN:
            cv::medianBlur(src, dst, params.median_ksize);
            break;
        case FilterType::BILATERAL:
            cv::bilateralFilter(src, dst, params.bilateral_d,
                                params.bilateral_sigma_color, params.bilateral_sigma_space);
            break;
        case FilterType::NONE:
        default:
            if (dst.data != src.data) {
                src.copyTo(dst);
            }
            break;
    }
    return true;
}

bool applyFilterGPU(const cv::Mat& src, cv::Mat& dst, const FilterParams& params) {
    LOG_ERROR("GPU filter not implemented yet, falling back to CPU");
    return applyFilterCPU(src, dst, params);
}

} // namespace stereo_depth::census
