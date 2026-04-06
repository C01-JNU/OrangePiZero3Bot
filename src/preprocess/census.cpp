// Copyright (C) 2026 C01-JNU
// SPDX-License-Identifier: GPL-3.0-only
//
// This file is part of FishTotem.
//
// FishTotem is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FishTotem is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FishTotem. If not, see <https://www.gnu.org/licenses/>.


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
                   int win_w, int win_h, int thresh, TransformType type)
        : m_src(src), m_dst(dst), m_win_w(win_w), m_win_h(win_h), m_thresh(thresh), m_type(type) {}

    virtual void operator()(const cv::Range& range) const override {
        int half_w = m_win_w / 2;
        int half_h = m_win_h / 2;

        if (m_type == TransformType::CENSUS) {
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
        } else { // RANK
            for (int y = range.start; y < range.end; ++y) {
                for (int x = half_w; x < m_src.cols - half_w; ++x) {
                    uchar center = m_src.at<uchar>(y, x);
                    uint16_t count = 0;
                    for (int dy = -half_h; dy <= half_h; ++dy) {
                        for (int dx = -half_w; dx <= half_w; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            uchar neighbor = m_src.at<uchar>(y + dy, x + dx);
                            int diff = static_cast<int>(neighbor) - center;
                            if (m_thresh == 0) {
                                if (neighbor < center) ++count;
                            } else {
                                if (abs(diff) > m_thresh) {
                                    if (neighbor < center) ++count;
                                }
                            }
                        }
                    }
                    m_dst.at<uint16_t>(y, x) = count;
                }
            }
        }
    }

private:
    const cv::Mat& m_src;
    cv::Mat& m_dst;
    int m_win_w, m_win_h;
    int m_thresh;
    TransformType m_type;
};

CensusTransform::CensusTransform() = default;
CensusTransform::~CensusTransform() = default;

bool CensusTransform::init(const CensusParams& params) {
    m_params = params;
    if (m_params.window_width % 2 == 0 || m_params.window_height % 2 == 0) {
        LOG_ERROR("窗口尺寸必须为奇数");
        return false;
    }
    LOG_INFO("CensusTransform 初始化: 窗口 {}x{}, 自适应阈值={}, 变换类型={}",
             m_params.window_width, m_params.window_height, m_params.adaptive_threshold,
             (m_params.transform_type == TransformType::CENSUS ? "Census" : "Rank"));
    return true;
}

bool CensusTransform::compute(const cv::Mat& left, const cv::Mat& right,
                              cv::Mat& left_census, cv::Mat& right_census) {
    if (m_params.transform_type == TransformType::CENSUS) {
        computeOneCensus(left, left_census);
        computeOneCensus(right, right_census);
    } else {
        computeOneRank(left, left_census);
        computeOneRank(right, right_census);
    }
    return true;
}

void CensusTransform::computeOneCensus(const cv::Mat& src, cv::Mat& dst) {
    int half_w = m_params.window_width / 2;
    int half_h = m_params.window_height / 2;
    dst.create(src.size(), CV_16U);
    dst.setTo(cv::Scalar(0));

    cv::Range rows(half_h, src.rows - half_h);
    CensusParallel body(src, dst,
                        m_params.window_width, m_params.window_height,
                        m_params.adaptive_threshold, TransformType::CENSUS);
    cv::parallel_for_(rows, body);
}

void CensusTransform::computeOneRank(const cv::Mat& src, cv::Mat& dst) {
    int half_w = m_params.window_width / 2;
    int half_h = m_params.window_height / 2;
    dst.create(src.size(), CV_16U);
    dst.setTo(cv::Scalar(0));

    cv::Range rows(half_h, src.rows - half_h);
    CensusParallel body(src, dst,
                        m_params.window_width, m_params.window_height,
                        m_params.adaptive_threshold, TransformType::RANK);
    cv::parallel_for_(rows, body);
}

} // namespace stereo_depth::preprocess
