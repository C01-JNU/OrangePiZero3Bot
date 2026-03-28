#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OrangePi Zero3 立体视觉上位机（硬件监控版）
功能：
- 左侧2x2网格显示：左校正图（流0）、左眼Census图（流1）、右眼Census图（流2）、原始拼接图（流3）
- 右侧面板：硬件状态列表 + 折线图（CPU温度、CPU占用率）
- 全局保存开关和目录选择
- 基端口可动态修改
- 保存图片自动分文件夹：stream<id>/YYYY-MM-DD/
"""

import socket
import struct
import threading
import queue
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import os
import time
from collections import defaultdict, deque
from datetime import datetime   # 新增用于日期文件夹

# ========== 默认配置 ==========
DEFAULT_BASE_PORT = 5000
BUFFER_SIZE = 65536

STREAM_LEFT = 0      # 左校正图
STREAM_DISPARITY = 1 # 左眼Census图
STREAM_DEPTH = 2     # 右眼Census图
STREAM_STITCHED = 3  # 原始拼接图
STREAM_HARDWARE = 4  # 硬件状态

# ========== UDP接收器 ==========
class UdpReceiver:
    def __init__(self, stream_id, port):
        self.stream_id = stream_id
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        self.fragments = defaultdict(dict)
        self.lock = threading.Lock()
        self.callbacks = []

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.settimeout(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"流 {self.stream_id} 监听端口 {self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.sock:
            self.sock.close()

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65536)
                self._process_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"流 {self.stream_id} 接收错误: {e}")

    def _process_packet(self, data):
        header_fmt = '<IHHHHQ'
        header_size = struct.calcsize(header_fmt)
        if len(data) < header_size:
            return
        header = data[:header_size]
        payload = data[header_size:]
        frame_id, stream_id, packet_idx, total_packets, data_len, timestamp = struct.unpack(header_fmt, header)
        if stream_id != self.stream_id:
            return
        if len(payload) != data_len:
            return
        with self.lock:
            frag_dict = self.fragments[frame_id]
            if 'total' not in frag_dict:
                frag_dict['total'] = total_packets
                frag_dict['packets'] = {}
                frag_dict['timestamp'] = timestamp
            frag_dict['packets'][packet_idx] = payload
            if len(frag_dict['packets']) == frag_dict['total']:
                full_data = b''.join(frag_dict['packets'][i] for i in sorted(frag_dict['packets'].keys()))
                del self.fragments[frame_id]
                for cb in self.callbacks:
                    cb(self.stream_id, frame_id, timestamp, full_data)
            if len(self.fragments) > 100:
                sorted_frames = sorted(self.fragments.keys())
                for old_fid in sorted_frames[:-50]:
                    del self.fragments[old_fid]


# ========== 图像解码 ==========
def decode_image_from_bytes(data):
    try:
        header_fmt = 'iii'
        header_size = struct.calcsize(header_fmt)
        if len(data) < header_size:
            return None
        rows, cols, img_type = struct.unpack(header_fmt, data[:header_size])
        img_data = data[header_size:]

        # OpenCV 类型映射 (CV_8U, CV_8UC3, CV_16U, CV_16S, CV_32S, CV_64F)
        type_map = {
            0: (np.uint8, 1),   # CV_8UC1
            16: (np.uint8, 3),  # CV_8UC3
            2: (np.uint16, 1),  # CV_16UC1
            3: (np.int16, 1),   # CV_16SC1
            4: (np.int32, 1),   # CV_32SC1
            6: (np.float64, 1), # CV_64FC1
        }
        if img_type not in type_map:
            print(f"不支持的图像类型: {img_type}")
            return None
        dtype, channels = type_map[img_type]

        expected_size = rows * cols * channels * np.dtype(dtype).itemsize
        if len(img_data) != expected_size:
            print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(img_data)}")
            return None

        arr = np.frombuffer(img_data, dtype=dtype).reshape(rows, cols, channels)
        if channels == 1:
            arr = arr.squeeze()
        return arr
    except Exception as e:
        print(f"解码图像失败: {e}")
        return None


# ========== 硬件状态解码 ==========
def decode_hardware_status(data):
    """解析 HardwareStatus 二进制数据"""
    # 格式：timestamp(uint64) + 8个float + uptime(uint64) + flags(uint8)
    # 总字节：8 + 8*4 + 8 + 1 = 49
    if len(data) < 49:
        return None
    offset = 0
    timestamp = struct.unpack_from('<Q', data, offset)[0]; offset += 8
    cpu_temp = struct.unpack_from('<f', data, offset)[0]; offset += 4
    gpu_temp = struct.unpack_from('<f', data, offset)[0]; offset += 4
    ddr_temp = struct.unpack_from('<f', data, offset)[0]; offset += 4
    cpu_usage = struct.unpack_from('<f', data, offset)[0]; offset += 4
    mem_used = struct.unpack_from('<f', data, offset)[0]; offset += 4
    mem_total = struct.unpack_from('<f', data, offset)[0]; offset += 4
    swap_used = struct.unpack_from('<f', data, offset)[0]; offset += 4
    swap_total = struct.unpack_from('<f', data, offset)[0]; offset += 4
    uptime = struct.unpack_from('<Q', data, offset)[0]; offset += 8
    flags = struct.unpack_from('<B', data, offset)[0]

    cpu_valid = (flags & 1) != 0
    gpu_valid = (flags & 2) != 0
    ddr_valid = (flags & 4) != 0

    return {
        'timestamp': timestamp,
        'cpu_temp': cpu_temp if cpu_valid else None,
        'gpu_temp': gpu_temp if gpu_valid else None,
        'ddr_temp': ddr_temp if ddr_valid else None,
        'cpu_usage': cpu_usage,
        'mem_used': mem_used,
        'mem_total': mem_total,
        'swap_used': swap_used,
        'swap_total': swap_total,
        'uptime': uptime,
    }


# ========== 接收器管理器（支持动态端口修改）==========
class ReceiverManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.receivers = {}
                    cls._instance.base_port = DEFAULT_BASE_PORT
        return cls._instance

    def set_base_port(self, port):
        self.stop_all()
        self.base_port = port

    def get_receiver(self, stream_id):
        port = self.base_port + stream_id
        if stream_id not in self.receivers:
            recv = UdpReceiver(stream_id, port)
            recv.start()
            self.receivers[stream_id] = recv
        return self.receivers[stream_id]

    def stop_all(self):
        for recv in self.receivers.values():
            recv.stop()
        self.receivers.clear()


# ========== 嵌入式视图（用于主窗口左侧）==========
class EmbeddedView:
    def __init__(self, parent, stream_id, name, global_save_var, global_dir_var):
        self.parent = parent
        self.stream_id = stream_id
        self.name = name
        self.global_save_var = global_save_var
        self.global_dir_var = global_dir_var

        self.frame = ttk.LabelFrame(parent, text=f"{name} (流{stream_id})")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.info_label = ttk.Label(self.frame, text="FPS: --")
        self.info_label.pack(anchor=tk.NW, padx=5, pady=2)

        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.receiver = ReceiverManager().get_receiver(stream_id)
        self.receiver.add_callback(self.on_frame)

        self.fps_counter = deque(maxlen=30)
        self.last_frame_id = None
        self.current_img = None

        self.update_fps_display()

    def on_frame(self, stream_id, frame_id, timestamp, data):
        img = decode_image_from_bytes(data)
        if img is None:
            return
        self.last_frame_id = frame_id
        self.current_img = img
        self.fps_counter.append(time.time())

        if self.global_save_var.get():
            self.save_image(img, frame_id, timestamp)

        self.frame.after(0, lambda: self.update_display(img))

    def update_display(self, img):
        display_img = self.prepare_display(img)
        pil_img = Image.fromarray(display_img)
        pil_img.thumbnail((400, 300))
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

    def prepare_display(self, img):
        """将各种类型的图像转换为 RGB 8位用于显示"""
        if len(img.shape) == 2:  # 单通道
            # 已经是 8 位灰度
            if img.dtype == np.uint8:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # 其他整数或浮点类型：归一化到 0-255
            else:
                # 处理可能存在的无效区域（如边界零值）
                if np.issubdtype(img.dtype, np.integer):
                    # 整数类型：直接使用全图范围归一化
                    minv, maxv = img.min(), img.max()
                else:
                    # 浮点类型：忽略 NaN
                    valid = ~np.isnan(img)
                    if np.any(valid):
                        minv, maxv = img[valid].min(), img[valid].max()
                    else:
                        minv, maxv = 0, 1
                if maxv > minv:
                    norm = ((img.astype(np.float32) - minv) / (maxv - minv) * 255).clip(0, 255).astype(np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        else:  # 三通道（BGR）
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def save_image(self, img, frame_id, timestamp):
        base_dir = self.global_dir_var.get()
        if not base_dir:
            return

        # 按流ID和日期分文件夹
        date_str = datetime.now().strftime("%Y-%m-%d")
        subdir = os.path.join(base_dir, f"stream{self.stream_id}", date_str)
        os.makedirs(subdir, exist_ok=True)

        filename = f"stream{self.stream_id}_{frame_id}_{timestamp}.png"
        filepath = os.path.join(subdir, filename)

        if len(img.shape) == 2:
            cv2.imwrite(filepath, img)
        else:
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def update_fps_display(self):
        if len(self.fps_counter) >= 2:
            times = list(self.fps_counter)
            if times[-1] > times[0]:
                fps = (len(times)-1) / (times[-1] - times[0])
                self.info_label.config(text=f"FPS: {fps:.1f}")
        self.frame.after(1000, self.update_fps_display)

    def destroy(self):
        self.receiver.remove_callback(self.on_frame)
        self.frame.destroy()


# ========== 硬件监控面板（右侧）==========
class HardwarePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # 历史数据存储（最多60个点）
        self.max_points = 60
        self.cpu_temp_history = deque(maxlen=self.max_points)
        self.cpu_usage_history = deque(maxlen=self.max_points)
        self.time_history = deque(maxlen=self.max_points)  # 相对时间（秒）

        self.current_status = {}

        # 创建接收器监听硬件流
        self.receiver = ReceiverManager().get_receiver(STREAM_HARDWARE)
        self.receiver.add_callback(self.on_hardware_frame)

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        # 列表显示区域
        list_frame = ttk.LabelFrame(self, text="当前硬件状态", padding=5)
        list_frame.pack(fill=tk.X, padx=5, pady=5)

        self.list_vars = {}
        labels = [
            ("CPU 温度", "cpu_temp", "℃"),
            ("GPU 温度", "gpu_temp", "℃"),
            ("DDR 温度", "ddr_temp", "℃"),
            ("CPU 占用", "cpu_usage", "%"),
            ("内存使用", "mem_used", "MB"),
            ("内存总量", "mem_total", "MB"),
            ("交换使用", "swap_used", "MB"),
            ("交换总量", "swap_total", "MB"),
            ("系统运行时间", "uptime", "秒"),
        ]
        for i, (name, key, unit) in enumerate(labels):
            row_frame = ttk.Frame(list_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=f"{name}:", width=12, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value="--")
            self.list_vars[key] = var
            ttk.Label(row_frame, textvariable=var, width=15, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=unit, width=5, anchor=tk.W).pack(side=tk.LEFT)

        # 折线图区域
        plot_frame = ttk.LabelFrame(self, text="实时趋势 (最近60秒)", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(plot_frame, bg='white', height=200)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def on_hardware_frame(self, stream_id, frame_id, timestamp, data):
        status = decode_hardware_status(data)
        if status is None:
            return
        self.current_status = status

        # 更新列表
        self.list_vars['cpu_temp'].set(f"{status['cpu_temp']:.1f}" if status['cpu_temp'] is not None else "N/A")
        self.list_vars['gpu_temp'].set(f"{status['gpu_temp']:.1f}" if status['gpu_temp'] is not None else "N/A")
        self.list_vars['ddr_temp'].set(f"{status['ddr_temp']:.1f}" if status['ddr_temp'] is not None else "N/A")
        self.list_vars['cpu_usage'].set(f"{status['cpu_usage']:.1f}")
        self.list_vars['mem_used'].set(f"{status['mem_used']:.1f}")
        self.list_vars['mem_total'].set(f"{status['mem_total']:.1f}")
        self.list_vars['swap_used'].set(f"{status['swap_used']:.1f}" if status['swap_used'] is not None else "N/A")
        self.list_vars['swap_total'].set(f"{status['swap_total']:.1f}" if status['swap_total'] is not None else "N/A")
        self.list_vars['uptime'].set(f"{status['uptime']}")

        # 更新历史（使用时间戳的相对值）
        if status['cpu_temp'] is not None:
            self.cpu_temp_history.append(status['cpu_temp'])
        else:
            self.cpu_temp_history.append(None)
        self.cpu_usage_history.append(status['cpu_usage'])

        # 计算相对时间（以第一个点的时间为0）
        if len(self.time_history) == 0:
            self.time_history.append(0)
        else:
            last_time = self.time_history[-1]
            # 假设数据到达间隔接近发送间隔，简单递增1秒
            self.time_history.append(last_time + 1)

    def update_plot(self):
        self.draw_plot()
        self.after(1000, self.update_plot)

    def draw_plot(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 10 or h <= 10:
            return

        margin = 30
        plot_w = w - 2*margin
        plot_h = h - 2*margin

        # 画坐标轴
        self.canvas.create_line(margin, h-margin, w-margin, h-margin, fill='black')  # X轴
        self.canvas.create_line(margin, margin, margin, h-margin, fill='black')      # Y轴

        # 如果没有数据
        if len(self.cpu_temp_history) == 0:
            return

        # 确定Y轴范围
        all_values = []
        for v in self.cpu_temp_history:
            if v is not None:
                all_values.append(v)
        all_values.extend(self.cpu_usage_history)
        if not all_values:
            return
        y_min = min(all_values)
        y_max = max(all_values)
        if y_max - y_min < 0.1:
            y_min = 0
            y_max = 100
        y_range = y_max - y_min

        # 绘制 CPU 温度曲线（红色）
        pts = []
        for i, val in enumerate(self.cpu_temp_history):
            if val is None:
                continue
            x = margin + (i / (self.max_points-1)) * plot_w
            y = h - margin - ((val - y_min) / y_range) * plot_h
            pts.append((x, y))
        if len(pts) > 1:
            for j in range(len(pts)-1):
                self.canvas.create_line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1], fill='red', width=2)

        # 绘制 CPU 占用曲线（蓝色）
        pts = []
        for i, val in enumerate(self.cpu_usage_history):
            x = margin + (i / (self.max_points-1)) * plot_w
            y = h - margin - ((val - y_min) / y_range) * plot_h
            pts.append((x, y))
        if len(pts) > 1:
            for j in range(len(pts)-1):
                self.canvas.create_line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1], fill='blue', width=2)

        # 添加图例
        self.canvas.create_text(margin+20, margin+10, text="CPU温度", fill='red', anchor=tk.W)
        self.canvas.create_text(margin+20, margin+30, text="CPU占用", fill='blue', anchor=tk.W)

        # 添加Y轴标签
        self.canvas.create_text(margin-10, margin, text=f"{y_max:.0f}", anchor=tk.E)
        self.canvas.create_text(margin-10, h-margin, text=f"{y_min:.0f}", anchor=tk.E)


# ========== 图像显示窗口（弹窗）==========
class ImageWindow(tk.Toplevel):
    def __init__(self, parent, stream_id, port, title=None):
        super().__init__(parent)
        self.stream_id = stream_id
        self.port = port
        self.title(title or f"流 {stream_id}  (端口 {port})")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.save_enabled = tk.BooleanVar(value=False)
        self.save_dir = tk.StringVar(value=os.path.expanduser("~/stereo_captures"))
        self.last_frame_id = None
        self.current_img = None

        self.fps_counter = deque(maxlen=30)
        self.fps = 0.0

        self.create_widgets()
        self.receiver = ReceiverManager().get_receiver(stream_id)
        self.receiver.add_callback(self.on_frame)
        self.update_fps_display()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(control_frame, text="保存图像", variable=self.save_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="选择目录", command=self.choose_save_dir).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, textvariable=self.save_dir).pack(side=tk.LEFT, padx=5)
        self.fps_label = ttk.Label(control_frame, text="FPS: --")
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        self.image_label = ttk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def choose_save_dir(self):
        dirname = filedialog.askdirectory(initialdir=self.save_dir.get())
        if dirname:
            self.save_dir.set(dirname)

    def on_frame(self, stream_id, frame_id, timestamp, data):
        img = decode_image_from_bytes(data)
        if img is None:
            return
        self.last_frame_id = frame_id
        self.current_img = img
        self.fps_counter.append(time.time())
        if self.save_enabled.get():
            self.save_image(img, frame_id, timestamp)
        self.after(0, lambda: self.update_display(img))

    def update_display(self, img):
        display_img = self.prepare_display(img)
        pil_img = Image.fromarray(display_img)
        pil_img.thumbnail((600, 600))
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk
        self.title(f"流 {self.stream_id} (端口 {self.port}) 帧:{self.last_frame_id}")

    def prepare_display(self, img):
        """与 EmbeddedView 相同的显示处理"""
        if len(img.shape) == 2:
            if img.dtype == np.uint8:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                if np.issubdtype(img.dtype, np.integer):
                    minv, maxv = img.min(), img.max()
                else:
                    valid = ~np.isnan(img)
                    if np.any(valid):
                        minv, maxv = img[valid].min(), img[valid].max()
                    else:
                        minv, maxv = 0, 1
                if maxv > minv:
                    norm = ((img.astype(np.float32) - minv) / (maxv - minv) * 255).clip(0, 255).astype(np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def save_image(self, img, frame_id, timestamp):
        base_dir = self.save_dir.get()
        if not base_dir:
            return

        # 按流ID和日期分文件夹
        date_str = datetime.now().strftime("%Y-%m-%d")
        subdir = os.path.join(base_dir, f"stream{self.stream_id}", date_str)
        os.makedirs(subdir, exist_ok=True)

        filename = f"stream{self.stream_id}_{frame_id}_{timestamp}.png"
        filepath = os.path.join(subdir, filename)

        if len(img.shape) == 2:
            cv2.imwrite(filepath, img)
        else:
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def update_fps_display(self):
        if len(self.fps_counter) >= 2:
            times = list(self.fps_counter)
            if times[-1] > times[0]:
                fps = (len(times)-1) / (times[-1] - times[0])
                self.fps = fps
                self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.after(1000, self.update_fps_display)

    def on_close(self):
        self.receiver.remove_callback(self.on_frame)
        self.destroy()


# ========== 主应用程序 ==========
class StereoViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OrangePi Zero3 立体视觉上位机（硬件监控版）")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 全局保存变量
        self.global_save_enabled = tk.BooleanVar(value=False)
        self.global_save_dir = tk.StringVar(value=os.path.expanduser("~/stereo_captures"))

        # 基端口变量
        self.base_port_var = tk.StringVar(value=str(DEFAULT_BASE_PORT))

        # 创建菜单
        self.create_menu()

        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 使用 PanedWindow 分割左右区域
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 左侧区域（控制面板 + 2x2图像网格）
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)  # 左侧占3份

        # 顶部控制面板
        self.create_control_panel(left_frame)

        # 图像显示区域（2x2网格）
        display_frame = ttk.Frame(left_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i in range(2):
            display_frame.rowconfigure(i, weight=1)
            display_frame.columnconfigure(i, weight=1)

        # 创建四个嵌入式视图
        self.views = []
        view_configs = [
            (STREAM_LEFT, "左校正图"),
            (STREAM_DISPARITY, "左眼Census图"),
            (STREAM_DEPTH, "右眼Census图"),
            (STREAM_STITCHED, "原始拼接图")
        ]
        positions = [(0,0), (0,1), (1,0), (1,1)]
        for (sid, name), (row, col) in zip(view_configs, positions):
            frame = ttk.Frame(display_frame)
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            view = EmbeddedView(frame, sid, name,
                                self.global_save_enabled, self.global_save_dir)
            self.views.append(view)

        # 右侧区域（硬件监控面板）
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)  # 右侧占1份
        self.hardware_panel = HardwarePanel(right_frame)
        self.hardware_panel.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, padx=5, pady=2)

        self.manager = ReceiverManager()

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="全局控制", padding=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 第一行：保存选项
        save_frame = ttk.Frame(control_frame)
        save_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(save_frame, text="保存所有图像",
                        variable=self.global_save_enabled).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="选择保存目录",
                   command=self.choose_global_dir).pack(side=tk.LEFT, padx=5)
        ttk.Label(save_frame, textvariable=self.global_save_dir).pack(side=tk.LEFT, padx=5)

        # 第二行：基端口设置
        port_frame = ttk.Frame(control_frame)
        port_frame.pack(fill=tk.X, pady=2)
        ttk.Label(port_frame, text="基端口:").pack(side=tk.LEFT, padx=5)
        port_entry = ttk.Entry(port_frame, textvariable=self.base_port_var, width=10)
        port_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(port_frame, text="应用", command=self.apply_base_port).pack(side=tk.LEFT, padx=5)
        ttk.Button(port_frame, text="清空队列", command=self.clear_queues).pack(side=tk.RIGHT, padx=5)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开新窗口", command=self.open_new_window)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def choose_global_dir(self):
        dirname = filedialog.askdirectory(initialdir=self.global_save_dir.get())
        if dirname:
            self.global_save_dir.set(dirname)

    def apply_base_port(self):
        try:
            new_port = int(self.base_port_var.get())
            if new_port < 1024 or new_port > 65535:
                raise ValueError("端口范围1024-65535")
        except Exception as e:
            messagebox.showerror("错误", f"无效端口号: {e}")
            return

        ReceiverManager().set_base_port(new_port)
        # 销毁现有视图（将重新创建）
        for view in self.views:
            view.destroy()
        self.views.clear()
        # 重新创建左侧视图（略，此处简化，可重启程序）
        self.status_var.set(f"基端口已更新为 {new_port}，请重启程序以使全部视图生效")
        messagebox.showinfo("提示", "基端口已更新，建议重启程序")

    def clear_queues(self):
        self.status_var.set("队列已清空（仅显示）")

    def open_new_window(self):
        stream_id = simpledialog.askinteger("新窗口", "请输入流ID:", parent=self.root, minvalue=0, maxvalue=255)
        if stream_id is None:
            return
        port = ReceiverManager().base_port + stream_id
        win = ImageWindow(self.root, stream_id, port)

    def show_about(self):
        messagebox.showinfo("关于", "OrangePi Zero3 立体视觉上位机\n左侧2x2图像 + 右侧硬件监控\n支持硬件状态列表与趋势图")

    def on_closing(self):
        ReceiverManager().stop_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StereoViewerApp(root)
    root.mainloop()