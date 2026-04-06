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


#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

namespace stereo_depth {
namespace utils {

/**
 * @brief 硬件状态数据结构
 */
struct HardwareStatus {
    uint64_t timestamp;          // 采集时间戳（微秒）
    float cpu_temp;              // CPU 温度（摄氏度）
    float gpu_temp;              // GPU 温度（摄氏度）
    float ddr_temp;              // DDR 温度（摄氏度）
    float cpu_usage_percent;     // CPU 总体占用率（0~100）
    float memory_used_mb;        // 已用内存（MB）
    float memory_total_mb;       // 总内存（MB）
    float swap_used_mb;          // 已用交换空间（MB）
    float swap_total_mb;         // 总交换空间（MB）
    uint64_t uptime_seconds;     // 系统启动时间（秒）

    // 有效性标志
    bool cpu_temp_valid;
    bool gpu_temp_valid;
    bool ddr_temp_valid;

    HardwareStatus() { reset(); }

    void reset() {
        timestamp = 0;
        cpu_temp = 0.0f;
        gpu_temp = 0.0f;
        ddr_temp = 0.0f;
        cpu_usage_percent = 0.0f;
        memory_used_mb = 0.0f;
        memory_total_mb = 0.0f;
        swap_used_mb = 0.0f;
        swap_total_mb = 0.0f;
        uptime_seconds = 0;
        cpu_temp_valid = false;
        gpu_temp_valid = false;
        ddr_temp_valid = false;
    }
};

/**
 * @brief 硬件监控器（独立线程采集）
 */
class HardwareMonitor {
public:
    HardwareMonitor();
    ~HardwareMonitor();

    // 禁止拷贝
    HardwareMonitor(const HardwareMonitor&) = delete;
    HardwareMonitor& operator=(const HardwareMonitor&) = delete;

    // 允许移动
    HardwareMonitor(HardwareMonitor&&) = default;
    HardwareMonitor& operator=(HardwareMonitor&&) = default;

    /**
     * @brief 初始化监控器（从配置读取参数）
     * @return 成功返回 true
     */
    bool initialize();

    /**
     * @brief 启动采集线程
     * @return 成功返回 true
     */
    bool start();

    /**
     * @brief 停止采集线程（阻塞等待线程结束）
     */
    void stop();

    /**
     * @brief 获取最新一次采集的硬件状态（线程安全）
     */
    HardwareStatus getLatestStatus() const;

    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return m_running.load(); }

private:
    // 采集循环
    void collectLoop();

    // 采集单次状态（由 collectLoop 调用）
    HardwareStatus collectOnce();

    // 读取系统文件辅助函数
    static float readTemperatureFromFile(const std::string& path);
    float readCpuUsage();  // 改为非静态成员函数
    static bool readMemoryInfo(float& mem_total, float& mem_used, float& swap_total, float& swap_used);
    static uint64_t readUptime();

    // 上一次采集的 CPU 时间数据（用于计算占用率）
    struct CpuTime {
        unsigned long long user;
        unsigned long long nice;
        unsigned long long system;
        unsigned long long idle;
        unsigned long long iowait;
        unsigned long long irq;
        unsigned long long softirq;
        unsigned long long steal;
        unsigned long long guest;
        unsigned long long guest_nice;
    };
    mutable std::mutex m_cpu_time_mutex;
    CpuTime m_prev_cpu_time;

    // 配置参数
    double m_interval_seconds;    // 采集间隔（秒）
    bool m_enable_gpu;            // 是否尝试读取 GPU 温度
    bool m_enable_ddr;            // 是否尝试读取 DDR 温度

    // 线程控制
    std::atomic<bool> m_running;
    std::thread m_collect_thread;
    mutable std::mutex m_status_mutex;
    HardwareStatus m_latest_status;
};

} // namespace utils
} // namespace stereo_depth
