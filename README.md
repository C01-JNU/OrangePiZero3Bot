# OrangePi Zero3 双目立体视觉系统

本项目代码由 **DeepSeek** 编写，为香橙派 Zero3 的 **Ubuntu 24.04** 系统打造，专注于 **CPU 立体视觉处理（SGBM/BM）**。

> 香橙派 Zero3 GPU 算力有限，实测 GPU 模式性能远低于 CPU，故仅保留 CPU 模块。

系统支持实时图像采集、立体校正、视差计算、深度图生成，并通过 UDP 网络将图像流传输至 PC 端进行可视化与调试。采用纯 CPU 计算，无 ROS 依赖，模块化设计便于扩展。

## 功能特性

- **摄像头驱动**：支持 CHUSEI 3D 摄像头和 Mock 模式（从测试图像读取）
- **立体标定**：提供完整的标定工具，生成 YAML 标定文件
- **立体校正**：支持 RAW、CROP_ONLY、SCALE_TO_FIT 三种校正模式
- **CPU 立体匹配**：支持 SGBM 和 BM 算法，参数可配置
- **网络传输**：轻量级 UDP 分片传输，支持多流（左校正图、视差图、深度图、原始拼接图等）
- **PC 上位机**：Python Tkinter 图形界面，实时显示多路图像，支持保存和 FPS 显示
- **辅助工具**：提供图像分割脚本，便于准备标定图像

## 目录结构

```
.
├── CMakeLists.txt                # 顶层 CMake 配置
├── README.md                     # 本文件
├── docs/
│   ├──模块编写规范.txt
│   └──代码规范.txt
├── udp_viewer.py                  # PC 端上位机程序
├── config/                        # 配置文件目录
│   ├── calibration.yaml
│   ├── camera.yaml
│   ├── network.yaml               # 网络传输配置（IP、端口、启用开关）
│   ├── performance.yaml
│   ├── stereo.yaml
│   └── system.yaml
├── calibration_results/           # 标定结果存放目录（由标定程序生成）
├── images/                        # 图像目录
│   ├── calibration/               # 标定用图像（需用户放置）
│   ├── test/                      # 测试图像（Mock 模式使用）
│   └── output/                    # 程序运行输出目录（保存视差图、深度图等）
├── include/                       # 头文件
│   ├── calibration/
│   ├── camera/
│   ├── cpu_stereo/
│   ├── network/
│   └── utils/
├── src/                           # 源代码
│   ├── calibration/
│   ├── camera/
│   ├── cpu_stereo/
│   ├── network/
│   ├── utils/
│   ├── main.cpp                   # 实时处理主程序
│   ├── test.cpp                   # 测试程序（支持实时/批量+网络传输）
│   └── calibration/               # 标定相关主程序
└── tools/                         # 辅助脚本
    ├── chusei_cam_init.sh         # CHUSEI 摄像头初始化脚本
    └── split_stereo_images.sh     # 图像分割工具（将拼接图像分为左右图）
```

## 硬件要求

- **香橙派 Zero3**（运行 Ubuntu 24.04 或兼容系统）
- **双目 USB 摄像头**（如 CHUSEI 3D WebCam）
- **PC 机**（用于运行上位机，Windows/Linux 均可）

## 依赖安装（香橙派端）

### 系统依赖
```bash
sudo apt update
sudo apt install build-essential cmake git libopencv-dev libspdlog-dev libyaml-cpp-dev imagemagick
```

> `imagemagick` 用于分割脚本，如果不需要该脚本可跳过。

### 编译项目
```bash
mkdir build && cd build
cmake ..
make -j4
```

编译后生成的可执行文件位于 `build/bin/` 目录下：
- `stereo_depth`：实时处理主程序（含摄像头采集、校正、匹配、网络发送）
- `test_stereo_depth`：多功能测试程序（支持实时/批量处理，可选择是否使用网络）
- `stereo_calibrator`：立体标定工具
- `test_calibration`：校正测试工具

## 配置文件说明

所有配置文件位于 `config/` 目录下，重要参数如下：

- **`camera.yaml`**：摄像头参数（分辨率、驱动类型 `chusei`/`mock`）
- **`calibration.yaml`**：标定参数（棋盘格尺寸、是否启用校正、标定文件路径）
- **`stereo.yaml`**：立体匹配算法参数（算法选择 `sgbm`/`bm`，视差范围等）
- **`network.yaml`**：网络传输配置
  ```yaml
  network:
    ip: "192.168.1.10"        # PC 端 IP 地址
    port: 5000                  # 基端口（流 0 使用 port，流 1 使用 port+1，依此类推）
    enabled: true               # 是否启用网络传输
    max_fps: 0                  # 最大发送帧率（0 表示不限）
  ```
- **`system.yaml`**：系统日志级别等

## 使用流程

### 1. 准备标定图像（可选，但强烈建议）
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

### 2. 立体标定（首次使用必须执行）
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
编辑 `config/network.yaml`，将 `ip` 改为 PC 的 IP 地址，确保 `enabled: true`。  
**注意**：PC 端需放行对应端口的 UDP 入站（例如 5000-5010），详见下文。

### 4. 运行实时程序（香橙派端）
```bash
cd build/bin
./test_stereo_depth
```
根据提示选择模式：
- `1`：实时摄像头采集
- `2`：处理测试图像目录（`images/test`）

输入是否使用立体校正（1 是，0 否）。  
程序启动后将按配置进行采集、校正、匹配，并通过 UDP 发送图像流（流分配如下）：
- 流 0：左校正图
- 流 1：视差图
- 流 2：深度图
- 流 3：原始拼接图

### 5. PC 端上位机使用

上位机程序 `udp_viewer.py` 位于项目根目录，用于实时接收并显示香橙派传来的图像。

#### 安装 Python 依赖
```bash
pip install opencv-python numpy pillow
```

#### 运行上位机
```bash
python udp_viewer.py
```

#### 界面功能
- **2x2 网格显示**：左上角左校正图（流0）、右上角原始拼接图（流3）、左下角视差图（流1）、右下角深度图（流2）。
- **全局控制**：
  - 勾选“保存所有图像”后，接收到的图像会自动保存到指定目录（默认 `~/stereo_captures`）。
  - 可修改“基端口”（应与香橙派 `network.yaml` 中的 `port` 一致），点击“应用”后立即生效。
- **FPS 显示**：每个图像区域左上角显示当前帧率。
- **菜单**：可通过“文件→打开新窗口”查看其他流（例如流4、5等），需输入流 ID 和端口（基端口+流ID）。

#### 防火墙设置
确保 PC 防火墙允许 UDP 入站端口范围（例如 5000-5010）：
- 在 PowerShell（管理员）中执行：
  ```powershell
  New-NetFirewallRule -DisplayName "Allow OrangePi UDP" -Direction Inbound -Protocol UDP -LocalPort 5000-5010 -Action Allow
  ```
  或通过“高级安全 Windows Defender 防火墙”手动添加入站规则。

### 6. 停止程序
在香橙派端按 `Ctrl+C` 即可停止。上位机关闭窗口即可。

## 辅助工具

### 图像分割脚本 `split_stereo_images.sh`
用于将拼接的立体图像（左右图并排）分割为左右眼独立图像，方便标定程序读取。

**依赖**：ImageMagick（`convert` 和 `identify` 命令）  
安装方式：`sudo apt install imagemagick`

**用法**：
```bash
./tools/split_stereo_images.sh <输入目录> <输出目录> [起始编号]
```
- `<输入目录>`：存放原始拼接图像的目录，支持 `.jpg`、`.jpeg`、`.png`
- `<输出目录>`：分割后图像保存目录（脚本会自动创建）
- `[起始编号]`：可选，输出文件名的起始数字（默认 0），例如起始编号 0 将生成 `000_left.jpg`、`000_right.jpg`……

**示例**：
```bash
cd tools
./split_stereo_images.sh ~/stereo_raw ../images/calibration 10
```
该命令会将 `~/stereo_raw` 中的所有图像分割，并以 `010_left.jpg`、`010_right.jpg`…… 的格式保存到 `images/calibration`。

**输出命名规则**：`<起始编号+索引>_left.jpg` 和 `<起始编号+索引>_right.jpg`，索引从 0 开始递增。

### CHUSEI 摄像头初始化脚本 `chusei_cam_init.sh`
用于向 CHUSEI 3D 摄像头发送 USB 控制命令，切换为双目模式。该脚本在程序运行时会自动调用（若摄像头驱动为 `chusei`），一般无需手动执行。

## 常见问题

### 上位机收不到数据
- 检查香橙派与 PC 是否在同一网络，IP 配置是否正确。
- 检查防火墙是否放行了 UDP 端口。

### 网络传输卡顿或丢包
- 可降低 `network.max_fps` 值。
- 使用有线网络可获得更稳定带宽。

### 分割脚本报错 “宽度为奇数”
- 输入图像宽度必须为偶数，否则无法平均分割为左右图。请确保原始图像宽度是偶数。

## 扩展开发

本项目采用模块化设计，添加新模块只需：
1. 在 `include/` 和 `src/` 下创建对应目录和文件。
2. 在 `src/` 下添加 `CMakeLists.txt`。
3. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory` 并链接库。

若需增加新的图像流发送，只需在 `test.cpp` 的实时循环中调用 `streamer.sendFrame(新流ID, 图像)`，上位机即可通过“打开新窗口”查看。
