#include "census/census_transform.h"
#include "utils/logger.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

namespace stereo_depth::census {

class CensusParallel : public cv::ParallelLoopBody {
public:
    CensusParallel(const cv::Mat& src, cv::Mat& dst, int win_w, int win_h,
                   CensusMode mode, int adaptive_threshold)
        : m_src(src), m_dst(dst), m_win_w(win_w), m_win_h(win_h),
          m_mode(mode), m_thresh(adaptive_threshold) {}

    virtual void operator()(const cv::Range& range) const override {
        int half_w = m_win_w / 2;
        int half_h = m_win_h / 2;

        for (int y = range.start; y < range.end; ++y) {
            for (int x = half_w; x < m_src.cols - half_w; ++x) {
                uint16_t code = 0;
                int bit = 0;

                if (m_mode == CensusMode::ADAPTIVE) {
                    uchar center = m_src.at<uchar>(y, x);
                    for (int dy = -half_h; dy <= half_h; ++dy) {
                        for (int dx = -half_w; dx <= half_w; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            uchar neighbor = m_src.at<uchar>(y + dy, x + dx);
                            int diff = static_cast<int>(neighbor) - center;
                            if (diff > m_thresh) {
                                code |= (1 << bit);
                            } // else if (diff < -m_thresh) 编码为0，不做操作
                            ++bit;
                        }
                    }
                } else { // STANDARD
                    uchar center = m_src.at<uchar>(y, x);
                    for (int dy = -half_h; dy <= half_h; ++dy) {
                        for (int dx = -half_w; dx <= half_w; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            uchar neighbor = m_src.at<uchar>(y + dy, x + dx);
                            if (neighbor < center) {
                                code |= (1 << bit);
                            }
                            ++bit;
                        }
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
    CensusMode m_mode;
    int m_thresh;
};

bool CensusTransform::computeCPU(const cv::Mat& src, cv::Mat& dst) {
    int half_w = m_win_width / 2;
    int half_h = m_win_height / 2;

    dst.create(src.size(), CV_16U);
    dst.setTo(cv::Scalar(0));

    cv::Range rows(half_h, src.rows - half_h);
    CensusParallel body(src, dst, m_win_width, m_win_height, m_census_mode, m_adaptive_threshold);
    cv::parallel_for_(rows, body);

    return true;
}

} // namespace stereo_depth::census
