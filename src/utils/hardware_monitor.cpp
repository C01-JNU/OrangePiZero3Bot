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


#include "utils/hardware_monitor.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"

#include <fstream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/sysinfo.h>

namespace stereo_depth {
namespace utils {

HardwareMonitor::HardwareMonitor()
    : m_interval_seconds(1.0)
    , m_enable_gpu(true)
    , m_enable_ddr(true)
    , m_running(false)
{
    // 初始化上一次 CPU 时间
    std::memset(&m_prev_cpu_time, 0, sizeof(m_prev_cpu_time));
}

HardwareMonitor::~HardwareMonitor() {
    stop();
}

bool HardwareMonitor::initialize() {
    auto& cfg = ConfigManager::getInstance().getConfig();

    // 读取配置
    m_interval_seconds = cfg.get<double>("hardware_monitor.interval_seconds", 1.0);
    m_enable_gpu = cfg.get<bool>("hardware_monitor.enable_gpu", true);
    m_enable_ddr = cfg.get<bool>("hardware_monitor.enable_ddr", true);

    LOG_INFO("HardwareMonitor initialized: interval={}s, enable_gpu={}, enable_ddr={}",
             m_interval_seconds, m_enable_gpu, m_enable_ddr);
    return true;
}

bool HardwareMonitor::start() {
    if (m_running.load()) {
        LOG_WARN("HardwareMonitor already running");
        return false;
    }

    m_running = true;
    m_collect_thread = std::thread(&HardwareMonitor::collectLoop, this);

    LOG_INFO("HardwareMonitor started");
    return true;
}

void HardwareMonitor::stop() {
    if (m_running.load()) {
        m_running = false;
        if (m_collect_thread.joinable()) {
            m_collect_thread.join();
        }
        LOG_INFO("HardwareMonitor stopped");
    }
}

HardwareStatus HardwareMonitor::getLatestStatus() const {
    std::lock_guard<std::mutex> lock(m_status_mutex);
    return m_latest_status;
}

void HardwareMonitor::collectLoop() {
    while (m_running.load()) {
        HardwareStatus status = collectOnce();

        // 更新最新状态
        {
            std::lock_guard<std::mutex> lock(m_status_mutex);
            m_latest_status = status;
        }

        // 等待下一个采集周期
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(m_interval_seconds * 1000)));
    }
}

HardwareStatus HardwareMonitor::collectOnce() {
    HardwareStatus status;
    status.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // 读取温度
    if (m_enable_gpu) {
        status.gpu_temp = readTemperatureFromFile("/sys/class/thermal/thermal_zone2/temp");
        status.gpu_temp_valid = (status.gpu_temp >= 0);
    } else {
        status.gpu_temp_valid = false;
    }

    if (m_enable_ddr) {
        status.ddr_temp = readTemperatureFromFile("/sys/class/thermal/thermal_zone1/temp");
        status.ddr_temp_valid = (status.ddr_temp >= 0);
    } else {
        status.ddr_temp_valid = false;
    }

    // CPU 温度总是可用
    status.cpu_temp = readTemperatureFromFile("/sys/class/thermal/thermal_zone0/temp");
    status.cpu_temp_valid = (status.cpu_temp >= 0);

    // CPU 占用率
    status.cpu_usage_percent = readCpuUsage();
    if (status.cpu_usage_percent < 0) status.cpu_usage_percent = 0;

    // 内存信息
    float mem_total, mem_used, swap_total, swap_used;
    if (readMemoryInfo(mem_total, mem_used, swap_total, swap_used)) {
        status.memory_total_mb = mem_total;
        status.memory_used_mb = mem_used;
        status.swap_total_mb = swap_total;
        status.swap_used_mb = swap_used;
    }

    // 系统启动时间
    status.uptime_seconds = readUptime();

    return status;
}

float HardwareMonitor::readTemperatureFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_DEBUG("Failed to open temperature file: {}", path);
        return -1.0f;
    }
    int raw_temp = 0;
    file >> raw_temp;
    if (file.fail()) {
        LOG_DEBUG("Failed to parse temperature from {}", path);
        return -1.0f;
    }
    return static_cast<float>(raw_temp) / 1000.0f; // 内核返回毫摄氏度
}

float HardwareMonitor::readCpuUsage() {
    std::ifstream stat("/proc/stat");
    if (!stat.is_open()) {
        LOG_DEBUG("Failed to open /proc/stat");
        return -1.0f;
    }

    std::string line;
    std::getline(stat, line); // 第一行是 cpu 整体统计
    if (line.empty() || line.compare(0, 3, "cpu") != 0) {
        return -1.0f;
    }

    CpuTime curr;
    unsigned long long dummy;
    std::istringstream iss(line);
    std::string cpu_label;
    iss >> cpu_label
        >> curr.user
        >> curr.nice
        >> curr.system
        >> curr.idle
        >> curr.iowait
        >> curr.irq
        >> curr.softirq
        >> curr.steal
        >> curr.guest
        >> curr.guest_nice;
    if (iss.fail()) {
        return -1.0f;
    }

    // 如果 prev 未初始化，保存当前值并返回 0
    {
        std::lock_guard<std::mutex> lock(m_cpu_time_mutex);
        if (m_prev_cpu_time.user == 0 && m_prev_cpu_time.nice == 0 &&
            m_prev_cpu_time.system == 0 && m_prev_cpu_time.idle == 0) {
            m_prev_cpu_time = curr;
            return 0.0f;
        }

        unsigned long long prev_total = m_prev_cpu_time.user + m_prev_cpu_time.nice +
                                         m_prev_cpu_time.system + m_prev_cpu_time.idle +
                                         m_prev_cpu_time.iowait + m_prev_cpu_time.irq +
                                         m_prev_cpu_time.softirq + m_prev_cpu_time.steal +
                                         m_prev_cpu_time.guest + m_prev_cpu_time.guest_nice;
        unsigned long long curr_total = curr.user + curr.nice + curr.system +
                                         curr.idle + curr.iowait + curr.irq +
                                         curr.softirq + curr.steal +
                                         curr.guest + curr.guest_nice;
        unsigned long long total_diff = curr_total - prev_total;
        unsigned long long idle_diff = (curr.idle + curr.iowait) - (m_prev_cpu_time.idle + m_prev_cpu_time.iowait);

        m_prev_cpu_time = curr; // 更新

        if (total_diff == 0) return 0.0f;
        return 100.0f * (1.0f - static_cast<float>(idle_diff) / static_cast<float>(total_diff));
    }
}

bool HardwareMonitor::readMemoryInfo(float& mem_total, float& mem_used,
                                     float& swap_total, float& swap_used) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        LOG_DEBUG("Failed to open /proc/meminfo");
        return false;
    }

    std::string line;
    unsigned long long total_kb = 0, free_kb = 0, available_kb = 0;
    unsigned long long swap_total_kb = 0, swap_free_kb = 0;

    while (std::getline(meminfo, line)) {
        if (line.compare(0, 9, "MemTotal:") == 0) {
            std::sscanf(line.c_str(), "MemTotal: %llu kB", &total_kb);
        } else if (line.compare(0, 8, "MemFree:") == 0) {
            std::sscanf(line.c_str(), "MemFree: %llu kB", &free_kb);
        } else if (line.compare(0, 13, "MemAvailable:") == 0) {
            std::sscanf(line.c_str(), "MemAvailable: %llu kB", &available_kb);
        } else if (line.compare(0, 10, "SwapTotal:") == 0) {
            std::sscanf(line.c_str(), "SwapTotal: %llu kB", &swap_total_kb);
        } else if (line.compare(0, 9, "SwapFree:") == 0) {
            std::sscanf(line.c_str(), "SwapFree: %llu kB", &swap_free_kb);
        }
    }

    mem_total = static_cast<float>(total_kb) / 1024.0f;
    if (available_kb > 0) {
        mem_used = static_cast<float>(total_kb - available_kb) / 1024.0f;
    } else {
        // 如果没有 MemAvailable，使用 MemFree + Buffers + Cached 近似，但这里简单用 total - free
        mem_used = static_cast<float>(total_kb - free_kb) / 1024.0f;
    }

    swap_total = static_cast<float>(swap_total_kb) / 1024.0f;
    swap_used = static_cast<float>(swap_total_kb - swap_free_kb) / 1024.0f;
    return true;
}

uint64_t HardwareMonitor::readUptime() {
    std::ifstream uptime("/proc/uptime");
    if (!uptime.is_open()) {
        return 0;
    }
    double up_secs;
    uptime >> up_secs;
    if (uptime.fail()) return 0;
    return static_cast<uint64_t>(up_secs);
}

} // namespace utils
} // namespace stereo_depth
