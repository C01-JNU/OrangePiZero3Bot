#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "calibration/stereo_rectifier.hpp"
#include "calibration/calibration_loader.hpp"
#include "census/census_transform.h"
#include "camera/camera_factory.h"
#include "network/udp_streamer.h"
#include "utils/hardware_monitor.h"

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

bool isImageFile(const std::string& filename) {
    std::string ext;
    size_t pos = filename.rfind('.');
    if (pos != std::string::npos) {
        ext = filename.substr(pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
    }
    return false;
}

std::vector<std::string> listImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        LOG_ERROR("无法打开目录: {}", dir);
        return files;
    }
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

bool ensureDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) return true;
        LOG_ERROR("路径存在但不是目录: {}", path);
        return false;
    }
    if (mkdir(path.c_str(), 0755) == 0) return true;
    LOG_ERROR("创建目录失败: {}", path);
    return false;
}

void processFrame(const cv::Mat& left_orig, const cv::Mat& right_orig,
                  const cv::Mat& stitched_orig,
                  calibration::StereoRectifier* rectifier,
                  census::CensusTransform& census,
                  const std::string& out_orig_dir,
                  const std::string& prefix, int index,
                  bool save_images,
                  cv::Mat& out_left_rect, cv::Mat& out_right_rect,
                  cv::Mat& left_census, cv::Mat& right_census) {

    if (save_images) {
        std::string stitched_path = out_orig_dir + "/" + prefix + "_" + std::to_string(index) + "_stitched_orig.png";
        cv::imwrite(stitched_path, stitched_orig);
    }

    cv::Mat left_gray, right_gray;
    if (left_orig.channels() == 3) {
        cv::cvtColor(left_orig, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_orig, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left_orig.clone();
        right_gray = right_orig.clone();
    }

    if (rectifier && rectifier->isInitialized()) {
        if (!rectifier->rectifyPair(left_gray, right_gray, out_left_rect, out_right_rect)) {
            LOG_WARN("校正失败，跳过此帧");
            return;
        }
    } else {
        out_left_rect = left_gray;
        out_right_rect = right_gray;
    }

    if (!census.compute(out_left_rect, left_census)) {
        LOG_ERROR("左图 Census 变换失败");
        return;
    }
    if (!census.compute(out_right_rect, right_census)) {
        LOG_ERROR("右图 Census 变换失败");
        return;
    }

    if (save_images) {
        std::string left_census_path = out_orig_dir + "/" + prefix + "_" + std::to_string(index) + "_left_census.png";
        cv::imwrite(left_census_path, left_census);
        std::string right_census_path = out_orig_dir + "/" + prefix + "_" + std::to_string(index) + "_right_census.png";
        cv::imwrite(right_census_path, right_census);
        LOG_DEBUG("Census 图已保存: {} 和 {}", left_census_path, right_census_path);
    }
}

int main() {
    Logger::initialize("stereo_test", spdlog::level::info);
    LOG_INFO("=========================================");
    LOG_INFO("  OrangePiZero3-StereoDepth 测试程序 (Census 版)");
    LOG_INFO("=========================================");

    std::string exeDir = getExeDir();
    std::string configDir = exeDir + "/config";

    auto& cfg_mgr = ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig(configDir)) {
        LOG_ERROR("加载配置目录失败: {}", configDir);
        return -1;
    }
    const auto& cfg = cfg_mgr.getConfig();

    // 初始化网络传输模块
    network::UdpStreamer streamer;
    bool network_enabled = cfg.get<bool>("network.enabled", false);
    if (network_enabled) {
        if (!streamer.initFromConfig()) {
            LOG_WARN("网络传输初始化失败，将不使用网络发送");
        } else {
            streamer.start();
            LOG_INFO("网络传输已启动");
        }
    }

    // 硬件监控初始化
    HardwareMonitor hw_monitor;
    if (!hw_monitor.initialize()) {
        LOG_WARN("硬件监控初始化失败，将继续运行");
    } else {
        hw_monitor.start();
        LOG_INFO("硬件监控已启动");
    }
    double hw_send_interval = cfg.get<double>("hardware_monitor.send_interval", 1.0);
    int hw_stream_id = cfg.get<int>("hardware_monitor.stream_id", 4);
    auto last_hw_send = std::chrono::steady_clock::now();

    auto pack_hardware_status = [](const HardwareStatus& status) -> std::vector<uint8_t> {
        std::vector<uint8_t> buffer;
        auto append = [&](auto value) {
            auto ptr = reinterpret_cast<const uint8_t*>(&value);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(value));
        };
        append(status.timestamp);
        append(status.cpu_temp);
        append(status.gpu_temp);
        append(status.ddr_temp);
        append(status.cpu_usage_percent);
        append(status.memory_used_mb);
        append(status.memory_total_mb);
        append(status.swap_used_mb);
        append(status.swap_total_mb);
        append(status.uptime_seconds);
        uint8_t flags = 0;
        flags |= (status.cpu_temp_valid ? 1 : 0) << 0;
        flags |= (status.gpu_temp_valid ? 1 : 0) << 1;
        flags |= (status.ddr_temp_valid ? 1 : 0) << 2;
        append(flags);
        return buffer;
    };

    int mode = 0;
    std::cout << "\n请选择模式:\n";
    std::cout << "  1 - 使用真实摄像头实时采集\n";
    std::cout << "  2 - 处理测试图像目录 (images/test)\n";
    std::cout << "请输入数字 (1 或 2): ";
    std::cin >> mode;
    if (mode != 1 && mode != 2) {
        std::cerr << "无效选择，退出。\n";
        return -1;
    }

    int use_correction = 0;
    std::cout << "是否使用立体校正? (1: 是, 0: 否): ";
    std::cin >> use_correction;

    std::unique_ptr<calibration::StereoRectifier> rectifier = nullptr;
    double baseline = 0.0;
    double focal_length = 0.0;

    if (use_correction) {
        std::string calib_file = cfg.get<std::string>("calibration.calibration_file", "calibration_results/stereo_calibration.yml");
        std::string calibPath = exeDir + "/" + calib_file;
        rectifier = std::make_unique<calibration::StereoRectifier>();
        if (rectifier->loadAndInitialize(calibPath, calibration::RectificationMode::CROP_ONLY)) {
            LOG_INFO("立体校正器初始化成功 (CROP_ONLY)");
            auto params = rectifier->getCalibrationParams();
            baseline = params.baseline_meters;
            if (!params.camera_matrix_left.empty()) {
                focal_length = params.camera_matrix_left.at<double>(0, 0);
            }
            LOG_INFO("基线: {:.3f} m, 焦距: {:.1f} px", baseline, focal_length);
        } else {
            LOG_ERROR("立体校正器初始化失败，将不使用校正");
            rectifier = nullptr;
        }
    }

    // 初始化 Census 变换
    census::CensusTransform censusTransform;
    if (!censusTransform.initializeFromConfig()) {
        LOG_WARN("Census 从配置文件初始化失败，使用默认 5x5 窗口");
        censusTransform.initialize(5);
    }

    std::string base_out = exeDir + "/images/output";
    std::string out_orig = base_out + "/original";
    ensureDirectory(base_out);
    ensureDirectory(out_orig);

    if (mode == 1) {
        int cam_width = cfg.get<int>("camera.width", 640);
        int cam_height = cfg.get<int>("camera.height", 480);
        std::string camera_driver = cfg.get<std::string>("camera.driver", "chusei");
        int target_fps = cfg.get<int>("performance.target_fps", 10);

        auto cam = camera::CameraFactory::create(camera_driver);
        if (!cam || !cam->init(cam_width, cam_height, target_fps)) {
            LOG_ERROR("摄像头初始化失败");
            return -1;
        }
        LOG_INFO("摄像头已打开: {} ({}x{})", cam->getName(), cam->getWidth(), cam->getHeight());

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

        LOG_INFO("开始实时采集，按 Ctrl+C 停止...");

        while (running) {
            cv::Mat left, right;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (frame_queue.empty()) {
                    queue_cv.wait_for(lock, std::chrono::milliseconds(100));
                    if (frame_queue.empty()) continue;
                }
                auto& frame = frame_queue.front();
                left = frame.first;
                right = frame.second;
                frame_queue.pop();
            }

            cv::Mat stitched;
            cv::hconcat(left, right, stitched);

            cv::Mat left_rect, right_rect;
            cv::Mat left_census, right_census;
            bool save_images = !streamer.isRunning();
            processFrame(left, right, stitched,
                         rectifier.get(), censusTransform,
                         out_orig, "cam", frame_count,
                         save_images,
                         left_rect, right_rect,
                         left_census, right_census);

            if (streamer.isRunning()) {
                streamer.sendFrame(0, left_rect, frame_count);
                streamer.sendFrame(1, left_census, frame_count);
                streamer.sendFrame(2, right_census, frame_count);
                streamer.sendFrame(3, stitched, frame_count);
                LOG_DEBUG("网络发送帧 {}", frame_count);
            }

            if (hw_monitor.isRunning() && streamer.isRunning()) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_hw_send >= std::chrono::duration<double>(hw_send_interval)) {
                    auto status = hw_monitor.getLatestStatus();
                    auto data = pack_hardware_status(status);
                    streamer.sendData(hw_stream_id, data.data(), data.size());
                    last_hw_send = now;
                }
            }

            frame_count++;

            auto now = std::chrono::steady_clock::now();
            if (now < next_frame_time) {
                std::this_thread::sleep_until(next_frame_time);
            }
            next_frame_time += frame_duration;
        }

        capture_thread.join();

    } else {
        std::string test_dir = exeDir + "/images/test";
        std::vector<std::string> image_files = listImageFiles(test_dir);
        if (image_files.empty()) {
            LOG_ERROR("测试图像目录为空: {}", test_dir);
            return -1;
        }
        LOG_INFO("找到 {} 张测试图像", image_files.size());

        int index = 0;
        for (const auto& img_path : image_files) {
            LOG_INFO("处理: {}", img_path);
            cv::Mat stitched = cv::imread(img_path, cv::IMREAD_COLOR);
            if (stitched.empty()) {
                LOG_WARN("无法读取图像，跳过");
                continue;
            }

            int width = cfg.get<int>("camera.width", 640);
            int height = cfg.get<int>("camera.height", 480);
            cv::Mat resized;
            cv::resize(stitched, resized, cv::Size(width, height));

            int single_width = width / 2;
            cv::Mat left_orig = resized(cv::Rect(0, 0, single_width, height)).clone();
            cv::Mat right_orig = resized(cv::Rect(single_width, 0, single_width, height)).clone();

            cv::Mat left_rect, right_rect, left_census, right_census;
            processFrame(left_orig, right_orig, resized,
                         rectifier.get(), censusTransform,
                         out_orig, "test", index,
                         true,
                         left_rect, right_rect,
                         left_census, right_census);
            index++;
        }
        LOG_INFO("批量处理完成，共处理 {} 帧", index);
    }

    hw_monitor.stop();
    return 0;
}
