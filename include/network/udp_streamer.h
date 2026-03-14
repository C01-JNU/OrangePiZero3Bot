#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <opencv2/core.hpp>

namespace stereo_depth {
namespace network {

/**
 * @brief UDP分片传输器
 * 
 * 将图像数据或任意二进制数据分片并通过UDP发送到指定IP和端口。
 * 支持多流（通过不同端口区分），自动分片重组。
 */
class UdpStreamer {
public:
    UdpStreamer();
    ~UdpStreamer();

    // 禁止拷贝
    UdpStreamer(const UdpStreamer&) = delete;
    UdpStreamer& operator=(const UdpStreamer&) = delete;

    /**
     * @brief 初始化传输器
     * @param target_ip 目标IP地址
     * @param base_port 基础端口号（实际使用的端口为 base_port + stream_id）
     * @param max_fps 最大发送帧率（0表示不限）
     * @return 成功返回true
     */
    bool init(const std::string& target_ip, int base_port, int max_fps = 0);

    /**
     * @brief 从全局配置初始化
     * @return 成功返回true
     */
    bool initFromConfig();

    /**
     * @brief 发送一帧图像（阻塞当前线程直到数据入队）
     * @param stream_id 流ID（0:左图, 1:右图, 2:视差图, 3:深度图等）
     * @param image 图像数据（CV_8U 单通道或三通道）
     * @param timestamp 时间戳（可选，默认0）
     * @return 成功返回true
     */
    bool sendFrame(int stream_id, const cv::Mat& image, uint64_t timestamp = 0);

    /**
     * @brief 发送任意二进制数据
     * @param stream_id 流ID
     * @param data 数据指针
     * @param size 数据大小（字节）
     * @param timestamp 时间戳（可选，默认0）
     * @return 成功返回true
     */
    bool sendData(int stream_id, const void* data, size_t size, uint64_t timestamp = 0);

    /**
     * @brief 启动发送线程
     */
    void start();

    /**
     * @brief 停止发送线程
     */
    void stop();

    /**
     * @brief 检查是否正在运行
     */
    bool isRunning() const { return running_.load(); }

private:
    // 分片包结构
    struct Packet {
        uint32_t frame_id;      // 帧ID（递增）
        uint16_t stream_id;     // 流ID
        uint16_t packet_idx;    // 当前包序号（0开始）
        uint16_t total_packets; // 总包数
        uint16_t data_len;      // 本包数据长度
        uint64_t timestamp;     // 时间戳
        std::vector<uint8_t> data;
    };

    // 发送队列条目
    struct FrameItem {
        uint32_t frame_id;
        uint16_t stream_id;
        uint64_t timestamp;
        std::vector<uint8_t> data; // 完整数据（已序列化）
    };

    void sendThreadFunc();
    bool sendPacket(const Packet& pkt);
    std::vector<Packet> fragmentFrame(const FrameItem& item, size_t mtu_payload);

    // 网络相关
    int sockfd_ = -1;
    std::string target_ip_;
    int base_port_ = 5000;
    int max_fps_ = 0;
    std::atomic<bool> initialized_{false};

    // 线程控制
    std::thread send_thread_;
    std::atomic<bool> running_{false};
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<FrameItem> frame_queue_;
    std::atomic<uint32_t> next_frame_id_{1};

    // 统计
    std::atomic<uint64_t> sent_frames_{0};
    std::atomic<uint64_t> sent_packets_{0};
    std::atomic<uint64_t> dropped_frames_{0};
};

} // namespace network
} // namespace stereo_depth
