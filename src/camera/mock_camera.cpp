#include "camera/mock_camera.h"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include <unistd.h>
#include "utils/logger.hpp"

namespace stereo_depth::camera {

static bool isImageFile(const std::string& filename) {
    std::string ext;
    size_t pos = filename.rfind('.');
    if (pos != std::string::npos) {
        ext = filename.substr(pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
    }
    return false;
}

static std::vector<std::string> listImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;
    struct dirent* entry;
    while ((entry = readdir(dp))) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (isImageFile(name)) {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

MockCamera::MockCamera() = default;

MockCamera::~MockCamera() {
    if (m_running) {
        m_running = false;
        if (m_thread.joinable()) m_thread.join();
    }
}

bool MockCamera::init(int width, int height, int fps) {
    m_width = width;
    m_height = height;
    m_single_width = width / 2;
    m_fps = fps;

    // 获取可执行文件路径，构造 images/test 绝对路径
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::filesystem::path exePath(result);
        // 可执行文件在 bin 目录下，测试图像在 bin/images/test
        m_test_dir = (exePath.parent_path() / "images" / "test").string();
    } else {
        m_test_dir = "images/test";
    }
    
    LOG_INFO("模拟摄像头：测试图像目录 = {}", m_test_dir);

    m_image_files = listImageFiles(m_test_dir);
    if (m_image_files.empty()) {
        LOG_ERROR("模拟摄像头：测试图像目录为空: {}", m_test_dir);
        return false;
    }
    LOG_INFO("模拟摄像头：找到 {} 张测试图像", m_image_files.size());

    m_running = true;
    m_thread = std::thread(&MockCamera::captureThread, this);
    m_initialized = true;
    return true;
}

bool MockCamera::grab(cv::Mat& left, cv::Mat& right) {
    if (!m_initialized) return false;
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_frame_queue.empty()) {
        if (!m_cv.wait_for(lock, std::chrono::seconds(1), [this]{ return !m_frame_queue.empty(); })) {
            return false;
        }
    }
    auto& frame = m_frame_queue.front();
    left = frame.first.clone();
    right = frame.second.clone();
    m_frame_queue.pop();
    return true;
}

void MockCamera::captureThread() {
    auto frame_duration = std::chrono::milliseconds(1000 / m_fps);
    auto next_frame_time = std::chrono::steady_clock::now();

    while (m_running) {
        if (m_image_files.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        const std::string& img_path = m_image_files[m_current_index];
        m_current_index = (m_current_index + 1) % m_image_files.size();

        cv::Mat stitched = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (stitched.empty()) {
            LOG_WARN("模拟摄像头：无法读取图像 {}", img_path);
            std::this_thread::sleep_for(frame_duration);
            continue;
        }

        if (stitched.cols != m_width || stitched.rows != m_height) {
            cv::resize(stitched, stitched, cv::Size(m_width, m_height));
        }

        cv::Mat left = stitched(cv::Rect(0, 0, m_single_width, m_height)).clone();
        cv::Mat right = stitched(cv::Rect(m_single_width, 0, m_single_width, m_height)).clone();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_frame_queue.size() >= 2) {
                m_frame_queue.pop();
            }
            m_frame_queue.emplace(left, right);
        }
        m_cv.notify_one();

        next_frame_time += frame_duration;
        std::this_thread::sleep_until(next_frame_time);
    }
}

} // namespace stereo_depth::camera
