#include "camera/chusei_camera.h"
#include "utils/logger.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <dirent.h>
#include <regex>
#include <thread>
#include <chrono>
#include <filesystem>
#include <array>
#include <memory>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

namespace stereo_depth::camera {

namespace fs = std::filesystem;

static std::string executeCommand(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::string full_cmd = cmd + " 2>/dev/null";
    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(full_cmd.c_str(), "r"), pclose);
    if (!pipe) return "错误";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        result += buffer.data();
    return result;
}

static std::pair<int, std::string> executeCommandWithStatus(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {-1, "popen失败"};
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        result += buffer.data();
    int status = pclose(pipe);
    return {status, result};
}

static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

ChuseiCamera::ChuseiCamera() = default;
ChuseiCamera::~ChuseiCamera() {
    if (cap_.isOpened()) cap_.release();
}

std::string ChuseiCamera::detectDevice() {
    LOG_INFO("正在精确检测 3D 摄像头设备...");
    for (int i = 0; i < 10; ++i) {
        std::string dev = "/dev/video" + std::to_string(i);
        if (!fileExists(dev)) continue;
        std::string info = executeCommand("v4l2-ctl -d " + dev + " --info");
        std::string formats = executeCommand("v4l2-ctl -d " + dev + " --list-formats");
        if (info.find("ERROR") != std::string::npos) continue;
        bool has_3d = info.find("3D Webcam") != std::string::npos;
        bool has_uvc = info.find("uvcvideo") != std::string::npos;
        bool has_cap = info.find("Video Capture") != std::string::npos;
        bool has_yuyv = formats.find("YUYV") != std::string::npos;
        LOG_INFO("检查设备 {}: 3D Webcam={}, uvcvideo={}, 视频捕获={}, YUYV格式={}",
                 dev, (has_3d?"是":"否"), (has_uvc?"是":"否"), (has_cap?"是":"否"), (has_yuyv?"是":"否"));
        if (has_3d && has_uvc && has_cap && has_yuyv) {
            LOG_INFO("找到精确匹配的 3D 摄像头设备: {}", dev);
            return dev;
        }
    }
    LOG_ERROR("未找到符合条件的 3D 摄像头设备");
    return "";
}

bool ChuseiCamera::verifyDevice(const std::string& dev) {
    LOG_INFO("验证设备 {} 是否可用...", dev);
    if (!fileExists(dev)) {
        LOG_ERROR("设备不存在");
        return false;
    }
    if (system(("v4l2-ctl -d " + dev + " --info >/dev/null 2>&1").c_str()) != 0) {
        LOG_ERROR("设备不可读或无权限");
        return false;
    }
    LOG_INFO("设备验证通过");
    return true;
}

bool ChuseiCamera::runInitScript(const std::string& dev) {
    // 获取可执行文件路径
    std::string exe_path;
    char result[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", result, PATH_MAX);
    if (len != -1) {
        result[len] = '\0';
        exe_path = result;
        LOG_INFO("可执行文件路径: {}", exe_path);
    } else {
        LOG_ERROR("无法获取可执行文件路径");
    }

    std::vector<std::string> candidate_paths;

    // 候选1：可执行文件所在目录下的 tools/ 子目录
    if (!exe_path.empty()) {
        fs::path exe_dir = fs::path(exe_path).parent_path(); // build/bin/
        fs::path candidate = exe_dir / "tools" / "chusei_cam_init.sh";
        candidate_paths.push_back(candidate.string());
        
        // 候选2：可执行文件上一级目录下的 tools/（如果从 build/bin 运行）
        fs::path parent_dir = exe_dir.parent_path(); // build/
        fs::path candidate2 = parent_dir / "tools" / "chusei_cam_init.sh";
        candidate_paths.push_back(candidate2.string());
    }

    // 候选3：项目源码目录中的 tools/（开发时直接运行）
    fs::path cwd = fs::current_path();
    fs::path candidate3 = cwd / "tools" / "chusei_cam_init.sh";
    candidate_paths.push_back(candidate3.string());

    // 候选4：从可执行文件路径向上找到项目根目录（假设 build 在项目根下）
    if (!exe_path.empty()) {
        fs::path exe_dir = fs::path(exe_path).parent_path();
        // 如果 exe_dir 是 build/bin，则项目根在 build/..
        fs::path project_root = exe_dir.parent_path().parent_path(); // 上两级：build/bin/.. => build/.. => 项目根
        fs::path candidate4 = project_root / "tools" / "chusei_cam_init.sh";
        candidate_paths.push_back(candidate4.string());
    }

    std::string found_path;
    for (const auto& path : candidate_paths) {
        LOG_INFO("尝试脚本路径: {}", path);
        if (fs::exists(path)) {
            LOG_INFO("找到脚本: {}", path);
            found_path = path;
            break;
        }
    }

    if (found_path.empty()) {
        LOG_ERROR("错误: 在所有候选路径中都找不到 chusei_cam_init.sh 脚本");
        return false;
    }

    // 执行脚本
    std::string cmd = found_path + " " + dev + " 2>&1";
    auto [status, output] = executeCommandWithStatus(cmd);
    LOG_INFO("脚本执行输出:\n{}", output);
    if (status != 0) {
        LOG_ERROR("脚本执行失败，返回码: {}", status);
        return false;
    }
    return true;
}

bool ChuseiCamera::verifyStereoMode() {
    const int test_frames = 5;
    // 静默读取几帧，确保摄像头工作正常
    for (int i = 0; i < test_frames; ++i) {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            LOG_ERROR("无法读取第 {} 帧，双目模式验证失败", i+1);
            return false;
        }
        // 不输出任何信息
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return true;
}

bool ChuseiCamera::init(int width, int height, int fps) {
    width_ = width;
    height_ = height;
    fps_ = fps;

    device_path_ = detectDevice();
    if (device_path_.empty()) return false;
    if (!verifyDevice(device_path_)) return false;

    cap_.open(device_path_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        LOG_ERROR("无法打开摄像头设备");
        return false;
    }

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS, fps_);

    LOG_INFO("摄像头分辨率设置: {}x{}",
             (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH),
             (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 激活摄像头，丢弃初始帧（开始流传输）
    LOG_INFO("激活摄像头，丢弃初始帧...");
    for (int i = 0; i < 5; ++i) {
        cv::Mat frame;
        if (cap_.read(frame))
            LOG_INFO("成功读取第 {} 帧，尺寸: {}x{}", i+1, frame.cols, frame.rows);
        else
            LOG_INFO("读取第 {} 帧失败", i+1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 执行初始化脚本（此时摄像头保持打开且正在传输）
    LOG_INFO("切换到双目模式...");
    if (!runInitScript(device_path_)) {
        LOG_ERROR("摄像头模式切换失败");
        return false;
    }

    LOG_INFO("等待模式切换稳定 (2秒)...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    LOG_INFO("丢弃切换后的不稳定帧...");
    for (int i = 0; i < 5; ++i) {
        cv::Mat frame;
        cap_.read(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!verifyStereoMode()) {
        LOG_ERROR("双目模式验证失败");
        return false;
    }

    initialized_ = true;
    LOG_INFO("摄像头初始化完成");
    return true;
}

bool ChuseiCamera::grab(cv::Mat& left, cv::Mat& right) {
    if (!initialized_) return false;
    cv::Mat frame;
    if (!cap_.read(frame)) return false;
    int half = frame.cols / 2;
    if (half <= 0) return false;
    cv::Mat l = frame(cv::Rect(0, 0, half, frame.rows));
    cv::Mat r = frame(cv::Rect(half, 0, half, frame.rows));
    if (frame.channels() == 3) {
        cv::cvtColor(l, left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(r, right, cv::COLOR_BGR2GRAY);
    } else {
        left = l.clone();
        right = r.clone();
    }
    return true;
}

} // namespace stereo_depth::camera
