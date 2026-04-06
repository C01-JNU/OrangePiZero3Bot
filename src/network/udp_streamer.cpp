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


#include "network/udp_streamer.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <chrono>
#include <thread>

namespace stereo_depth {
namespace network {

UdpStreamer::UdpStreamer() = default;

UdpStreamer::~UdpStreamer() {
    stop();
    if (sockfd_ != -1) {
        close(sockfd_);
    }
}

bool UdpStreamer::init(const std::string& target_ip, int base_port, int max_fps) {
    if (initialized_) return true;

    target_ip_ = target_ip;
    base_port_ = base_port;
    max_fps_ = max_fps;

    // 创建UDP socket
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR("创建socket失败");
        return false;
    }

    // 设置非阻塞（可选）
    int flags = fcntl(sockfd_, F_GETFL, 0);
    fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK);

    initialized_ = true;
    LOG_INFO("UDP流初始化完成，目标 {}:{}+stream_id, 最大帧率 {}", target_ip, base_port, max_fps);
    return true;
}

bool UdpStreamer::initFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    std::string ip = cfg.get<std::string>("network.ip", "127.0.0.1");
    int port = cfg.get<int>("network.port", 5000);
    int max_fps = cfg.get<int>("network.max_fps", 0);
    bool enabled = cfg.get<bool>("network.enabled", true);
    if (!enabled) {
        LOG_INFO("网络传输未启用");
        return false;
    }
    return init(ip, port, max_fps);
}

bool UdpStreamer::sendFrame(int stream_id, const cv::Mat& image, uint64_t timestamp) {
    if (!initialized_ || !running_) {
        LOG_WARN("流未启动，丢弃帧");
        return false;
    }

    // 序列化图像：将Mat转换为字节流（简单做法：存储尺寸、类型、数据）
    std::vector<uint8_t> buffer;
    // 头部：rows, cols, type
    int rows = image.rows;
    int cols = image.cols;
    int type = image.type();
    size_t data_size = image.total() * image.elemSize();
    buffer.resize(sizeof(rows) + sizeof(cols) + sizeof(type) + data_size);
    uint8_t* ptr = buffer.data();
    memcpy(ptr, &rows, sizeof(rows)); ptr += sizeof(rows);
    memcpy(ptr, &cols, sizeof(cols)); ptr += sizeof(cols);
    memcpy(ptr, &type, sizeof(type)); ptr += sizeof(type);
    memcpy(ptr, image.data, data_size);

    FrameItem item;
    item.frame_id = next_frame_id_++;
    item.stream_id = static_cast<uint16_t>(stream_id);
    item.timestamp = timestamp ? timestamp : 
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    item.data = std::move(buffer);

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= 10) { // 限制队列长度
            frame_queue_.pop();
            dropped_frames_++;
        }
        frame_queue_.push(std::move(item));
    }
    queue_cv_.notify_one();
    return true;
}

bool UdpStreamer::sendData(int stream_id, const void* data, size_t size, uint64_t timestamp) {
    if (!initialized_ || !running_) {
        LOG_WARN("流未启动，丢弃数据");
        return false;
    }

    // 直接拷贝二进制数据
    std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data),
                                 static_cast<const uint8_t*>(data) + size);

    FrameItem item;
    item.frame_id = next_frame_id_++;
    item.stream_id = static_cast<uint16_t>(stream_id);
    item.timestamp = timestamp ? timestamp : 
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    item.data = std::move(buffer);

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= 10) {
            frame_queue_.pop();
            dropped_frames_++;
        }
        frame_queue_.push(std::move(item));
    }
    queue_cv_.notify_one();
    return true;
}

void UdpStreamer::start() {
    if (!initialized_) {
        LOG_ERROR("未初始化，无法启动");
        return;
    }
    if (running_) return;
    running_ = true;
    send_thread_ = std::thread(&UdpStreamer::sendThreadFunc, this);
    LOG_INFO("UDP发送线程已启动");
}

void UdpStreamer::stop() {
    if (!running_) return;
    running_ = false;
    queue_cv_.notify_all();
    if (send_thread_.joinable()) send_thread_.join();
    LOG_INFO("UDP发送线程已停止，统计：发送帧 {}，发送包 {}，丢弃帧 {}",
             sent_frames_.load(), sent_packets_.load(), dropped_frames_.load());
}

void UdpStreamer::sendThreadFunc() {
    const size_t mtu_payload = 1400; // 以太网MTU 1500 - IP头部(20) - UDP头部(8) ≈ 1472，保守取1400
    sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;

    while (running_) {
        FrameItem item;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (frame_queue_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100));
                if (frame_queue_.empty()) continue;
            }
            item = std::move(frame_queue_.front());
            frame_queue_.pop();
        }

        // 帧率限制
        if (max_fps_ > 0) {
            static auto last_send_time = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_send_time).count();
            long min_interval_us = 1000000 / max_fps_;
            if (elapsed < min_interval_us) {
                std::this_thread::sleep_for(std::chrono::microseconds(min_interval_us - elapsed));
            }
            last_send_time = std::chrono::steady_clock::now();
        }

        // 分片
        auto packets = fragmentFrame(item, mtu_payload);
        dest_addr.sin_port = htons(base_port_ + item.stream_id);
        if (inet_pton(AF_INET, target_ip_.c_str(), &dest_addr.sin_addr) <= 0) {
            LOG_ERROR("无效IP地址: {}", target_ip_);
            continue;
        }

        bool all_sent = true;
        for (const auto& pkt : packets) {
            if (!sendPacket(pkt)) {
                all_sent = false;
                break;
            }
            sent_packets_++;
        }
        if (all_sent) {
            sent_frames_++;
        } else {
            dropped_frames_++;
        }
    }
}

std::vector<UdpStreamer::Packet> UdpStreamer::fragmentFrame(const FrameItem& item, size_t mtu_payload) {
    std::vector<Packet> packets;
    size_t total_data = item.data.size();
    uint16_t total_packets = (total_data + mtu_payload - 1) / mtu_payload;

    for (uint16_t i = 0; i < total_packets; ++i) {
        Packet pkt;
        pkt.frame_id = item.frame_id;
        pkt.stream_id = item.stream_id;
        pkt.packet_idx = i;
        pkt.total_packets = total_packets;
        pkt.timestamp = item.timestamp;

        size_t offset = i * mtu_payload;
        size_t len = std::min(mtu_payload, total_data - offset);
        pkt.data_len = len;
        pkt.data.assign(item.data.begin() + offset, item.data.begin() + offset + len);
        packets.push_back(std::move(pkt));
    }
    return packets;
}

bool UdpStreamer::sendPacket(const Packet& pkt) {
    // 构造发送缓冲区：头部 + 数据
    size_t header_size = sizeof(pkt.frame_id) + sizeof(pkt.stream_id) + 
                         sizeof(pkt.packet_idx) + sizeof(pkt.total_packets) + 
                         sizeof(pkt.data_len) + sizeof(pkt.timestamp);
    std::vector<uint8_t> buffer(header_size + pkt.data.size());
    uint8_t* ptr = buffer.data();
    memcpy(ptr, &pkt.frame_id, sizeof(pkt.frame_id)); ptr += sizeof(pkt.frame_id);
    memcpy(ptr, &pkt.stream_id, sizeof(pkt.stream_id)); ptr += sizeof(pkt.stream_id);
    memcpy(ptr, &pkt.packet_idx, sizeof(pkt.packet_idx)); ptr += sizeof(pkt.packet_idx);
    memcpy(ptr, &pkt.total_packets, sizeof(pkt.total_packets)); ptr += sizeof(pkt.total_packets);
    memcpy(ptr, &pkt.data_len, sizeof(pkt.data_len)); ptr += sizeof(pkt.data_len);
    memcpy(ptr, &pkt.timestamp, sizeof(pkt.timestamp)); ptr += sizeof(pkt.timestamp);
    memcpy(ptr, pkt.data.data(), pkt.data.size());

    sockaddr_in dest;
    dest.sin_family = AF_INET;
    dest.sin_port = htons(base_port_ + pkt.stream_id);
    if (inet_pton(AF_INET, target_ip_.c_str(), &dest.sin_addr) <= 0) {
        return false;
    }

    ssize_t sent = sendto(sockfd_, buffer.data(), buffer.size(), 0,
                           (struct sockaddr*)&dest, sizeof(dest));
    return sent == static_cast<ssize_t>(buffer.size());
}

} // namespace network
} // namespace stereo_depth
