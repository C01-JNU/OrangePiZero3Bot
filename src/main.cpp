#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unistd.h>

#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "calibration/calibration_loader.hpp"
#include "cpu_stereo/cpu_stereo_matcher.hpp"
#include "camera/camera_factory.h"

using namespace stereo_depth::utils;
using namespace stereo_depth;

std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        std::filesystem::path exePath(result);
        return exePath.parent_path().string();
    }
    return ".";
}

int main() {
    Logger::initialize("stereo_depth", spdlog::level::info);
    LOG_INFO("=========================================");
    LOG_INFO("  OrangePiZero3-StereoDepth 实时处理程序");
    LOG_INFO("=========================================");

    std::string exeDir = getExeDir();
    std::string configDir = exeDir + "/config";
    LOG_INFO("配置目录: {}", configDir);

    auto& cfg_mgr = ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig(configDir)) {
        LOG_ERROR("加载配置目录失败: {}", configDir);
        return -1;
    }
    const auto& cfg = cfg_mgr.getConfig();

    int target_fps = cfg.get<int>("performance.target_fps", 1);
    int cam_width = cfg.get<int>("camera.width", 1);
    int cam_height = cfg.get<int>("camera.height", 1);
    int single_width = cam_width / 2;
    std::string camera_driver = cfg.get<std::string>("camera.driver", "mock");

    auto cam = camera::CameraFactory::create(camera_driver);
    if (!cam) {
        LOG_ERROR("创建摄像头失败");
        return -1;
    }
    if (!cam->init(cam_width, cam_height, target_fps)) {
        LOG_ERROR("摄像头初始化失败");
        return -1;
    }
    LOG_INFO("摄像头已打开: {} ({}x{})", cam->getName(), cam->getWidth(), cam->getHeight());

    bool use_rectification = cfg.get<bool>("calibration.rectify_images", false);
    std::string calib_file = cfg.get<std::string>("calibration.calibration_file", "calibration_results/stereo_calibration.yml");
    std::string calibPath = exeDir + "/" + calib_file;
    std::unique_ptr<calibration::StereoRectifier> rectifier = nullptr;

    if (use_rectification) {
        calibration::CalibrationParams params;
        calibration::CalibrationLoader loader;
        if (loader.loadFromFile(calibPath, params)) {
            rectifier = std::make_unique<calibration::StereoRectifier>();
            if (rectifier->initialize(params, calibration::RectificationMode::SCALE_TO_FIT)) {
                LOG_INFO("立体校正器初始化成功");
            } else {
                LOG_ERROR("立体校正器初始化失败，将跳过校正");
                rectifier = nullptr;
            }
        } else {
            LOG_ERROR("加载标定文件失败，将跳过校正");
        }
    }

    // CPU立体匹配器初始化
    cpu_stereo::CpuStereoMatcher cpu_matcher;
    if (!cpu_matcher.initializeFromConfig()) {
        LOG_ERROR("CPU 匹配器初始化失败");
        return -1;
    }
    LOG_INFO("CPU 立体匹配引擎已初始化");

    std::queue<std::pair<cv::Mat, cv::Mat>> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running{true};

    std::thread capture_thread([&]() {
        while (running) {
            cv::Mat left, right;
            if (cam->grab(left, right)) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() >= 2) {
                    frame_queue.pop();
                }
                frame_queue.emplace(left.clone(), right.clone());
                queue_cv.notify_one();
            }
        }
    });

    auto frame_duration = std::chrono::milliseconds(1000 / target_fps);
    auto next_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    auto fps_start = std::chrono::steady_clock::now();

    LOG_INFO("开始处理，目标帧率: {} FPS", target_fps);

    std::string out_dir = exeDir + "/images/output";
    std::filesystem::create_directories(out_dir);

    while (running) {
        cv::Mat left, right;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (frame_queue.empty()) {
                queue_cv.wait_for(lock, std::chrono::milliseconds(100));
                if (frame_queue.empty()) continue;
            }
            while (frame_queue.size() > 1) {
                frame_queue.pop();
            }
            auto& frame = frame_queue.front();
            left = frame.first;
            right = frame.second;
            frame_queue.pop();
        }

        auto proc_start = std::chrono::high_resolution_clock::now();

        cv::Mat left_rect, right_rect;
        if (rectifier) {
            if (!rectifier->rectifyPair(left, right, left_rect, right_rect)) {
                LOG_WARN("校正失败，跳过此帧");
                continue;
            }
        } else {
            left_rect = left;
            right_rect = right;
        }

        cv::Mat disparity = cpu_matcher.compute(left_rect, right_rect);
        auto proc_end = std::chrono::high_resolution_clock::now();
        float proc_ms = std::chrono::duration<float, std::milli>(proc_end - proc_start).count();

        std::string out_path = out_dir + "/disparity_" + std::to_string(frame_count) + ".png";
        cv::Mat disp_8u;
        double min_val, max_val;
        cv::minMaxLoc(disparity, &min_val, &max_val);
        disparity.convertTo(disp_8u, CV_8U, 255.0 / (max_val > 0 ? max_val : 64.0));
        cv::imwrite(out_path, disp_8u);

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        float elapsed_sec = std::chrono::duration<float>(now - fps_start).count();
        if (elapsed_sec >= 5.0f) {
            float fps = frame_count / elapsed_sec;
            LOG_INFO("实际处理帧率: {:.2f} FPS (目标 {} FPS)", fps, target_fps);
            frame_count = 0;
            fps_start = now;
        }

        LOG_DEBUG("帧处理耗时: {:.2f} ms", proc_ms);

        auto now_time = std::chrono::steady_clock::now();
        if (now_time < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
        }
        next_frame_time += frame_duration;
    }

    capture_thread.join();
    LOG_INFO("程序结束");
    return 0;
}
