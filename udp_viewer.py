#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OrangePi Zero3 立体视觉上位机（增强版 v4）
功能：
- 主窗口2x2网格显示：左校正图（流0）、原始拼接图（流3）、视差图（流1）、深度图（流2）
- 每个窗口独立显示实时FPS
- 全局保存开关和目录选择
- 基端口可动态修改
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

# ========== 默认配置 ==========
DEFAULT_BASE_PORT = 5000
BUFFER_SIZE = 65536

STREAM_LEFT = 0      # 左校正图
STREAM_DISPARITY = 1 # 视差图
STREAM_DEPTH = 2     # 深度图
STREAM_STITCHED = 3  # 原始拼接图

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

        type_map = {
            0: (np.uint8, 1),   # CV_8UC1
            16: (np.uint8, 3),  # CV_8UC3
            2: (np.uint16, 1),  # CV_16UC1
            3: (np.int16, 1),   # CV_16SC1
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
        """修改基端口并重新启动所有接收器（需先停止所有）"""
        self.stop_all()
        self.base_port = port
        # 注意：接收器会在视图订阅时重新创建

    def get_receiver(self, stream_id):
        """获取指定流的接收器（使用当前基端口）"""
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


# ========== 嵌入式视图（用于主窗口）==========
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

        # 订阅接收器
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
        if len(img.shape) == 2:
            if img.dtype == np.uint8:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.dtype == np.uint16:
                valid = img > 0
                if np.any(valid):
                    minv, maxv = img[valid].min(), img[valid].max()
                    if maxv > minv:
                        norm = ((img.astype(np.float32)-minv)/(maxv-minv)*255).clip(0,255).astype(np.uint8)
                    else:
                        norm = np.zeros_like(img, dtype=np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
            elif img.dtype == np.int16:
                positive = img > 0
                if np.any(positive):
                    minv, maxv = img[positive].min(), img[positive].max()
                    if maxv > minv:
                        norm = ((img.astype(np.float32)-minv)/(maxv-minv)*255).clip(0,255).astype(np.uint8)
                    else:
                        norm = np.zeros_like(img, dtype=np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def save_image(self, img, frame_id, timestamp):
        save_dir = self.global_dir_var.get()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = f"stream{self.stream_id}_{frame_id}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
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
        """取消订阅并销毁框架"""
        self.receiver.remove_callback(self.on_frame)
        self.frame.destroy()


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
        self.receiver = ReceiverManager().get_receiver(stream_id)  # 使用管理器
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
        # 复用EmbeddedView的prepare_display
        if len(img.shape) == 2:
            if img.dtype == np.uint8:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.dtype == np.uint16:
                valid = img > 0
                if np.any(valid):
                    minv, maxv = img[valid].min(), img[valid].max()
                    if maxv > minv:
                        norm = ((img.astype(np.float32)-minv)/(maxv-minv)*255).clip(0,255).astype(np.uint8)
                    else:
                        norm = np.zeros_like(img, dtype=np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
            elif img.dtype == np.int16:
                positive = img > 0
                if np.any(positive):
                    minv, maxv = img[positive].min(), img[positive].max()
                    if maxv > minv:
                        norm = ((img.astype(np.float32)-minv)/(maxv-minv)*255).clip(0,255).astype(np.uint8)
                    else:
                        norm = np.zeros_like(img, dtype=np.uint8)
                else:
                    norm = np.zeros_like(img, dtype=np.uint8)
                return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def save_image(self, img, frame_id, timestamp):
        save_dir = self.save_dir.get()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = f"stream{self.stream_id}_{frame_id}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
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
        self.root.title("OrangePi Zero3 立体视觉上位机（2x2布局）")
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

        # 顶部控制面板（包含基端口设置）
        self.create_control_panel(main_frame)

        # 图像显示区域（2x2网格）
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 配置网格权重
        for i in range(2):
            display_frame.rowconfigure(i, weight=1)
            display_frame.columnconfigure(i, weight=1)

        # 创建四个嵌入式视图
        self.views = []
        view_configs = [
            (STREAM_LEFT, "左校正图"),
            (STREAM_STITCHED, "原始拼接图"),
            (STREAM_DISPARITY, "视差图"),
            (STREAM_DEPTH, "深度图")
        ]
        positions = [(0,0), (0,1), (1,0), (1,1)]  # 左上、右上、左下、右下
        for (sid, name), (row, col) in zip(view_configs, positions):
            frame = ttk.Frame(display_frame)
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            view = EmbeddedView(frame, sid, name,
                                self.global_save_enabled, self.global_save_dir)
            self.views.append(view)

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
        """应用新的基端口，重新初始化所有接收器"""
        try:
            new_port = int(self.base_port_var.get())
            if new_port < 1024 or new_port > 65535:
                raise ValueError("端口范围1024-65535")
        except Exception as e:
            messagebox.showerror("错误", f"无效端口号: {e}")
            return

        # 停止所有现有接收器
        ReceiverManager().stop_all()
        # 更新基端口
        ReceiverManager().base_port = new_port
        # 重新创建所有视图（需先销毁旧视图）
        for view in self.views:
            view.destroy()
        self.views.clear()

        # 重建视图
        display_frame = self.root.winfo_children()[1].winfo_children()[1]  # 获取显示区域的框架（略粗糙，但可用）
        view_configs = [
            (STREAM_LEFT, "左校正图"),
            (STREAM_STITCHED, "原始拼接图"),
            (STREAM_DISPARITY, "视差图"),
            (STREAM_DEPTH, "深度图")
        ]
        positions = [(0,0), (0,1), (1,0), (1,1)]
        for (sid, name), (row, col) in zip(view_configs, positions):
            frame = ttk.Frame(display_frame)
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            view = EmbeddedView(frame, sid, name,
                                self.global_save_enabled, self.global_save_dir)
            self.views.append(view)

        self.status_var.set(f"基端口已更新为 {new_port}")

    def clear_queues(self):
        self.status_var.set("队列已清空（仅显示）")

    def open_new_window(self):
        stream_id = simpledialog.askinteger("新窗口", "请输入流ID:", parent=self.root, minvalue=0, maxvalue=255)
        if stream_id is None:
            return
        port = ReceiverManager().base_port + stream_id
        win = ImageWindow(self.root, stream_id, port)

    def show_about(self):
        messagebox.showinfo("关于", "OrangePi Zero3 立体视觉上位机\n2x2布局\n动态基端口设置")

    def on_closing(self):
        ReceiverManager().stop_all()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StereoViewerApp(root)
    root.mainloop()