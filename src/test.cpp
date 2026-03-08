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
#include "cpu_stereo/cpu_stereo_matcher.hpp"
#include "camera/camera_factory.h"
#include "network/udp_streamer.h"  // 网络模块头文件

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

cv::Mat computeDepth16U(const cv::Mat& disparity, double baseline, double focal_length) {
    if (disparity.empty()) return cv::Mat();
    cv::Mat depth16u(disparity.size(), CV_16U, cv::Scalar(0));
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            short d = disparity.at<short>(y, x);
            if (d > 0) {
                float disp_real = d / 16.0f;
                float depth_m = static_cast<float>(baseline * focal_length / disp_real);
                int depth_mm = static_cast<int>(depth_m * 1000.0f + 0.5f);
                if (depth_mm > 0 && depth_mm < 65535) {
                    depth16u.at<uint16_t>(y, x) = static_cast<uint16_t>(depth_mm);
                }
            }
        }
    }
    return depth16u;
}

/**
 * @brief 处理一帧图像：校正、匹配、深度计算，并可选择保存结果
 * @param left_orig 左原始图像
 * @param right_orig 右原始图像
 * @param stitched_orig 拼接原始图像（用于保存原图）
 * @param rectifier 校正器指针
 * @param matcher 匹配器
 * @param baseline 基线长度（米）
 * @param focal_length 焦距（像素）
 * @param out_disparity_dir 视差图输出目录（仅当 save_images 为 true 时使用）
 * @param out_depth_dir 深度图输出目录
 * @param out_orig_dir 原图输出目录
 * @param prefix 文件名前缀
 * @param index 帧索引
 * @param save_images 是否保存图像到磁盘
 * @param out_left_rect 输出：校正后的左图
 * @param out_right_rect 输出：校正后的右图
 * @param out_disparity 输出：视差图（CV_16S）
 * @param out_depth16u 输出：深度图（CV_16U，毫米单位）
 */
void processFrame(const cv::Mat& left_orig, const cv::Mat& right_orig,
                  const cv::Mat& stitched_orig,
                  calibration::StereoRectifier* rectifier,
                  cpu_stereo::CpuStereoMatcher& matcher,
                  double baseline, double focal_length,
                  const std::string& out_disparity_dir,
                  const std::string& out_depth_dir,
                  const std::string& out_orig_dir,
                  const std::string& prefix, int index,
                  bool save_images,
                  cv::Mat& out_left_rect, cv::Mat& out_right_rect,
                  cv::Mat& out_disparity, cv::Mat& out_depth16u) {

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

    out_disparity = matcher.compute(out_left_rect, out_right_rect);
    if (out_disparity.empty()) {
        LOG_ERROR("视差计算失败");
        return;
    }

    if (save_images) {
        std::string disp_path = out_disparity_dir + "/" + prefix + "_" + std::to_string(index) + "_disp.png";
        cv::imwrite(disp_path, out_disparity);
    }

    if (baseline > 0 && focal_length > 0) {
        out_depth16u = computeDepth16U(out_disparity, baseline, focal_length);
        if (!out_depth16u.empty() && save_images) {
            std::string depth_path = out_depth_dir + "/" + prefix + "_" + std::to_string(index) + "_depth.png";
            cv::imwrite(depth_path, out_depth16u);
            LOG_DEBUG("深度图已保存: {}", depth_path);
        }
    } else {
        LOG_WARN("基线或焦距无效，无法计算深度图");
    }

    // 注释掉每帧的INFO日志，减少输出
    // LOG_INFO("帧 {} 处理完成，耗时 {:.2f} ms", index, matcher.getLastTimeMs());
}

int main() {
    Logger::initialize("stereo_test", spdlog::level::info);
    LOG_INFO("=========================================");
    LOG_INFO("  OrangePiZero3-StereoDepth 测试程序");
    LOG_INFO("=========================================");

    std::string exeDir = getExeDir();
    std::string configDir = exeDir + "/config";

    auto& cfg_mgr = ConfigManager::getInstance();
    if (!cfg_mgr.loadGlobalConfig(configDir)) {
        LOG_ERROR("加载配置目录失败: {}", configDir);
        return -1;
    }
    const auto& cfg = cfg_mgr.getConfig();

    // 初始化网络传输模块（从配置文件读取）
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
        calibration::CalibrationParams params;
        calibration::CalibrationLoader loader;
        if (loader.loadFromFile(calibPath, params)) {
            rectifier = std::make_unique<calibration::StereoRectifier>();
            if (rectifier->initialize(params, calibration::RectificationMode::SCALE_TO_FIT)) {
                LOG_INFO("立体校正器初始化成功");
                baseline = params.baseline_meters;
                if (!params.camera_matrix_left.empty()) {
                    focal_length = params.camera_matrix_left.at<double>(0, 0);
                }
                LOG_INFO("基线: {:.3f} m, 焦距: {:.1f} px", baseline, focal_length);
            } else {
                LOG_ERROR("立体校正器初始化失败，将不使用校正");
                rectifier = nullptr;
            }
        } else {
            LOG_ERROR("加载标定文件失败，将不使用校正");
        }
    }

    cpu_stereo::CpuStereoMatcher matcher;
    if (!matcher.initializeFromConfig()) {
        LOG_ERROR("CPU 匹配器初始化失败");
        return -1;
    }

    std::string base_out = exeDir + "/images/output";
    std::string out_disparity = base_out + "/disparity";
    std::string out_depth = base_out + "/depth";
    std::string out_orig = base_out + "/original";
    ensureDirectory(base_out);
    ensureDirectory(out_disparity);
    ensureDirectory(out_depth);
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

            cv::Mat left_rect, right_rect, disparity, depth16u;
            bool save_images = !streamer.isRunning();
            processFrame(left, right, stitched,
                         rectifier.get(), matcher,
                         baseline, focal_length,
                         out_disparity, out_depth, out_orig,
                         "cam", frame_count,
                         save_images,
                         left_rect, right_rect, disparity, depth16u);

            if (streamer.isRunning()) {
                streamer.sendFrame(0, left_rect, frame_count);
                streamer.sendFrame(1, disparity, frame_count);
                if (!depth16u.empty()) {
                    streamer.sendFrame(2, depth16u, frame_count);
                }
                // 新增：发送原始拼接图（流ID=3）
                streamer.sendFrame(3, stitched, frame_count);
                LOG_DEBUG("网络发送帧 {}", frame_count);
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
        // 批量处理模式：始终保存图片
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

            cv::Mat left_rect, right_rect, disparity, depth16u;
            processFrame(left_orig, right_orig, resized,
                         rectifier.get(), matcher,
                         baseline, focal_length,
                         out_disparity, out_depth, out_orig,
                         "test", index,
                         true,  // 始终保存
                         left_rect, right_rect, disparity, depth16u);
            index++;
        }
        LOG_INFO("批量处理完成，共处理 {} 帧", index);
    }

    return 0;
}
