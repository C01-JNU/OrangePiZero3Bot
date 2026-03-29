#include "preprocess/census.h"
#include "utils/logger.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

namespace stereo_depth::preprocess {

class CensusParallel : public cv::ParallelLoopBody {
public:
    CensusParallel(const cv::Mat& src, cv::Mat& dst,
                   int win_w, int win_h, int thresh)
        : m_src(src), m_dst(dst), m_win_w(win_w), m_win_h(win_h), m_thresh(thresh) {}

    virtual void operator()(const cv::Range& range) const override {
        int half_w = m_win_w / 2;
        int half_h = m_win_h / 2;

        for (int y = range.start; y < range.end; ++y) {
            for (int x = half_w; x < m_src.cols - half_w; ++x) {
                uchar center = m_src.at<uchar>(y, x);
                uint16_t code = 0;
                int bit = 0;
                for (int dy = -half_h; dy <= half_h; ++dy) {
                    for (int dx = -half_w; dx <= half_w; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        uchar neighbor = m_src.at<uchar>(y + dy, x + dx);
                        int diff = static_cast<int>(neighbor) - center;
                        if (m_thresh == 0) {
                            if (neighbor < center) code |= (1 << bit);
                        } else {
                            if (abs(diff) > m_thresh) {
                                if (neighbor < center) code |= (1 << bit);
                            }
                        }
                        ++bit;
                    }
                }
                m_dst.at<uint16_t>(y, x) = code;
            }
        }
    }

private:
    const cv::Mat& m_src;
    cv::Mat& m_dst;
    int m_win_w, m_win_h;
    int m_thresh;
};

CensusTransform::CensusTransform() = default;
CensusTransform::~CensusTransform() = default;

bool CensusTransform::init(const CensusParams& params) {
    m_params = params;
    if (m_params.window_width % 2 == 0 || m_params.window_height % 2 == 0) {
        LOG_ERROR("Census window size must be odd");
        return false;
    }
    LOG_INFO("CensusTransform initialized: window {}x{}, adaptive_threshold={}",
             m_params.window_width, m_params.window_height, m_params.adaptive_threshold);
    return true;
}

bool CensusTransform::compute(const cv::Mat& left, const cv::Mat& right,
                              cv::Mat& left_census, cv::Mat& right_census) {
    computeOne(left, left_census);
    computeOne(right, right_census);
    return true;
}

void CensusTransform::computeOne(const cv::Mat& src, cv::Mat& dst) {
    int half_w = m_params.window_width / 2;
    int half_h = m_params.window_height / 2;
    dst.create(src.size(), CV_16U);
    dst.setTo(cv::Scalar(0));

    cv::Range rows(half_h, src.rows - half_h);
    CensusParallel body(src, dst,
                        m_params.window_width, m_params.window_height,
                        m_params.adaptive_threshold);
    cv::parallel_for_(rows, body);
}

} // namespace stereo_depth::preprocess
