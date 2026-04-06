# FishTotem 双目视觉机器人系统

TotemFish 是一个面向香橙派 Zero3 的双目视觉机器人项目，提供双目图像采集、立体校正、Census 变换、硬件监控、UDP 网络传输及机器人控制接口。采用模块化设计，无 ROS 依赖，便于扩展和定制。

## 功能特性

- **摄像头驱动**：支持 CHUSEI 3D 摄像头和 Mock 模式（从测试图像读取）
- **立体标定**：提供完整的标定工具，生成 YAML 标定文件
- **立体校正**：支持 RAW、CROP_ONLY、SCALE_TO_FIT 三种校正模式，输出彩色校正图
- **Census 变换**：将彩色图转换为 Census 特征图（支持 CPU 和 GPU 两种后端，可通过配置切换）
- **硬件监控**：实时监控 CPU 温度、使用率、内存等信息，通过独立 UDP 流发送
- **网络传输**：轻量级 UDP 分片传输，支持多流：
  - 流 0：左校正图（彩色 BGR）
  - 流 1：左眼 Census 图（灰度）
  - 流 2：右眼 Census 图（灰度）
  - 流 3：原始拼接图（彩色）
  - 流 4：硬件状态（二进制）
- **机器人控制预留**：可扩展电机控制、传感器读取等模块
- **PC 上位机**：Python Tkinter 图形界面，实时显示多路图像及硬件趋势图，支持保存和 FPS 显示
- **辅助工具**：图像分割脚本，便于准备标定图像

## 硬件要求

- **香橙派 Zero3**（运行 Ubuntu 24.04 或兼容系统）
- **双目 USB 摄像头**（如 CHUSEI 3D WebCam）
- **PC 机**（用于运行上位机，Windows/Linux 均可）
- **可选：电机驱动板、传感器**（根据机器人扩展需求）

## 依赖安装（香橙派端）

### 系统依赖
```bash
sudo apt update
sudo apt install build-essential cmake git libopencv-dev libspdlog-dev libyaml-cpp-dev \
                 libopenblas-dev libarmadillo-dev
# 如需 GPU 加速 Census 变换，需安装 Vulkan 相关包
sudo apt install libvulkan-dev glslang-dev
```

> **注意**：即使前处理（去噪）使用 CPU，Census 变换仍可通过 Vulkan GPU 加速。若不需要 GPU 加速，可跳过 Vulkan 包的安装。

### 编译项目
```bash
mkdir build && cd build
cmake ..
make -j4
```

编译后生成的可执行文件位于 `build/bin/` 目录下：
- `start`：机器人主程序（整合视觉处理与运动控制）
- `test`：多功能测试程序（支持实时/批量处理、校正测试等）
- `stereo_calibrator`：立体标定工具
- `test_calibration`：校正测试工具

### 运行时环境变量（使用 GPU 加速时）
```bash
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1   # 启用 Mali GPU 的 Vulkan 驱动
```

## 配置文件说明

所有配置文件位于 `config/` 目录下，采用 YAML 格式，主要包含：
- `camera.yaml`：摄像头参数
- `calibration.yaml`：标定参数路径、棋盘格尺寸
- `network.yaml`：UDP 目标 IP、端口
- `preprocess.yaml`：Census 窗口大小、自适应阈值、是否使用 GPU 等
- `system.yaml`：日志级别、硬件监控配置

## 使用流程

### 1. 准备标定图像
如果你有拼接好的立体图像（左右图拼在一张图中），可使用 `tools/split_stereo_images.sh` 脚本将其分割为左右眼图像，并存入 `images/calibration/` 目录。

#### 分割脚本用法
```bash
cd tools
./split_stereo_images.sh <输入目录> <输出目录> [起始编号]
```
- `<输入目录>`：存放拼接图像的目录（例如 `~/stereo_raw`）
- `<输出目录>`：分割后的图像存放目录（通常为 `../images/calibration`）
- `[起始编号]`：可选，起始文件名编号（默认 0）

**示例**：
```bash
./split_stereo_images.sh ~/stereo_raw ../images/calibration 0
```
脚本要求：
- 系统已安装 ImageMagick（`convert` 命令）
- 输入图像宽度必须为偶数
- 输出格式：`000_left.jpg`、`000_right.jpg`、`001_left.jpg`……

分割完成后，即可进行标定。

### 2. 立体标定
确保 `images/calibration/` 目录中包含分割好的左右图像对（命名格式 `*_left.jpg` 和 `*_right.jpg`）。  
运行标定程序：
```bash
cd build/bin
./stereo_calibrator
```
标定结果将保存在 `calibration_results/` 目录下，包含：
- `stereo_calibration.yml`：标定参数文件
- `calibration_report.txt`：详细标定报告
- `rectification_validation_*.jpg`：校正验证图像

### 3. 配置网络
编辑 `config/network.yaml` 将 `ip` 改为 PC 的 IP 地址，确保 `enabled: true`。  
**注意**：PC 端需放行对应端口的 UDP 入站（例如 5000-5010），详见下文。

### 4. 运行测试程序（香橙派端）
```bash
cd build/bin
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1   # 可选，启用 GPU
./test
```
根据提示选择模式：
- `1`：实时摄像头采集
- `2`：处理测试图像目录（`images/test`）

输入是否使用立体校正（1 是，0 否）。  
程序启动后将按配置进行采集、校正、Census 变换，并通过 UDP 发送图像流：
- 流 0：左校正图（彩色 BGR）
- 流 1：左眼 Census 图（灰度）
- 流 2：右眼 Census 图（灰度）
- 流 3：原始拼接图（彩色）
- 流 4：硬件状态（二进制）

### 5. 运行主程序（含机器人控制）
```bash
cd build/bin
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1   # 可选，启用 GPU
./start
```
主程序在测试程序基础上增加了机器人控制接口（可扩展），启动后会持续运行，直到按 `Ctrl+C` 停止。

### 6. PC 端上位机使用

上位机程序 `udp_viewer.py` 位于项目根目录，支持图像显示和硬件监控。

#### 安装 Python 依赖
```bash
pip install opencv-python numpy pillow
```

#### 运行上位机
```bash
python udp_viewer.py
```

#### 界面功能
- **2x2 网格显示**：左校正图（彩色）、左 Census 图（灰度）、右 Census 图（灰度）、原始拼接图（彩色）
- **右侧硬件面板**：实时显示 CPU 温度、占用率、内存等，并绘制趋势图
- **全局控制**：可启用图像保存、修改基端口
- **FPS 显示**：每个图像区域左上角显示当前帧率

#### 防火墙设置
确保 PC 防火墙允许 UDP 入站端口范围（例如 5000-5010）：
- 在 PowerShell（管理员）中执行：
  ```powershell
  New-NetFirewallRule -DisplayName "Allow OrangePi UDP" -Direction Inbound -Protocol UDP -LocalPort 5000-5010 -Action Allow
  ```

### 7. 停止程序
在香橙派端按 `Ctrl+C` 停止。上位机关闭窗口即可。

## 辅助工具

### 图像分割脚本 `split_stereo_images.sh`
用于将拼接的立体图像分割为左右眼独立图像，方便标定程序读取。

**依赖**：ImageMagick（`convert` 和 `identify` 命令）  
安装方式：`sudo apt install imagemagick`

**用法**：
```bash
./tools/split_stereo_images.sh <输入目录> <输出目录> [起始编号]
```
- `<输入目录>`：存放原始拼接图像的目录，支持 `.jpg`、`.jpeg`、`.png`
- `<输出目录>`：分割后图像保存目录（脚本会自动创建）
- `[起始编号]`：可选，输出文件名的起始数字（默认 0）

**示例**：
```bash
cd tools
./split_stereo_images.sh ~/stereo_raw ../images/calibration 0
```
输出文件格式：`000_left.jpg`、`000_right.jpg`…… 索引从 0 递增。

### CHUSEI 摄像头初始化脚本 `chusei_cam_init.sh`
用于向 CHUSEI 3D 摄像头发送 USB 控制命令，切换为双目模式。该脚本在程序运行时会自动调用（若摄像头驱动为 `chusei`），一般无需手动执行。

## 常见问题

### 上位机收不到数据
- 检查香橙派与 PC 是否在同一网络，IP 配置是否正确。
- 检查防火墙是否放行了 UDP 端口。

### 校正后图像仍有黑边
确保使用 `CROP_ONLY` 模式（`test.cpp` 和 `start.cpp` 中已默认），且标定文件中包含准确的 `valid_roi_left/right`（通过 `stereo_calibrator` 生成）。

### 摄像头无法打开
- 检查 `/dev/video0` 是否存在，权限是否正确。
- 对于 CHUSEI 摄像头，确保已运行初始化脚本（程序会自动调用）。

### GPU 加速无法启用
- 确保已安装 `libvulkan-dev` 和 `glslang-dev`。
- 编译时 CMake 会自动检测 Vulkan，无需额外参数。
- 运行时必须设置环境变量 `export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1`。

## 扩展开发

本项目采用模块化设计，添加新模块只需：
1. 在 `include/` 和 `src/` 下创建对应目录和文件。
2. 在 `src/` 下添加 `CMakeLists.txt`。
3. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory` 并链接库。

若需增加新的图像流发送，只需在 `test.cpp` 或 `start.cpp` 的实时循环中调用 `streamer.sendFrame(新流ID, 图像)`，上位机即可通过“打开新窗口”查看。

机器人控制部分可在 `start.cpp` 中扩展，通过串口或 GPIO 与下位机通信。

## 项目改名说明

原项目名为 `OrangePiZero3-StereoDepth`，现已更名为 **TotemFish**，以体现其双目视觉机器人的完整定位。相关文档和代码注释正在逐步更新。
