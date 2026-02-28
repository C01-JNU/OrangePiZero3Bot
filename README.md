# OrangePiZero3Bot

本项目代码由 DeepSeek 编写，为香橙派 Zero3 的 Ubuntu 24.04 系统打造，专注于 CPU 立体视觉处理（SGBM/BM）。

> 香橙派 Zero3 GPU 算力有限，实测 GPU 模式性能远低于 CPU，故仅保留 CPU 模块。

## 功能
- **摄像头驱动**：支持 CHUSEI 3D 摄像头自动检测与初始化，也可使用模拟摄像头（图片循环）。
- **立体校正**：基于标定参数进行极线校正（RAW / CROP_ONLY / SCALE_TO_FIT）。
- **CPU 立体匹配**：SGBM / BM 算法，参数从配置文件读取。
- **标定工具**：棋盘格标定，生成含基线长度的 YAML 文件、报告及验证图。
- **测试程序**：批量处理测试图像，输出视差图。

## 目录结构
```
OrangePiZero3Bot/
├── include/               # 模块头文件
├── src/                   # 源文件（main.cpp, test.cpp 及各模块）
├── config/                # 多 YAML 配置文件（system, camera, calibration, stereo, performance）
├── tools/                 # 辅助脚本（初始化、分割图像、编译）
├── calibration_results/   # 标定结果存放
└── images/                # 图像目录（test, calibration, output）
```

## 依赖
```bash
sudo apt install libspdlog-dev libeigen3-dev libopencv-dev libyaml-cpp-dev
# CHUSEI 摄像头额外需：
sudo apt install v4l-utils uvcdynctrl
```

## 编译
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```
可执行文件生成在 `build/bin/`，配置文件和数据自动复制到该目录。

## 运行
- **实时处理（CHUSEI 摄像头）**：  
  `cd build/bin && ./stereo_depth`  
  视差图保存至 `images/output/`。

- **批量测试（图像文件）**：  
  将拼接图放入 `images/test/`，运行 `./test_stereo_depth`，视差图输出至 `images/output/`。

- **立体标定**：  
  1. 用 `tools/split_stereo_images.sh` 分割拼接图为左右对，放入 `images/calibration/`。  
  2. 确保 `config/calibration.yaml` 棋盘格参数正确，运行 `./stereo_calibrator`。  
  结果保存至 `calibration_results/`（含基线字段 `baseline_meters`）。

- **校正测试**：  
  `./test_calibration` 处理 `images/test/` 图像，校正结果存至 `images/calibrated/`。

## 添加自定义模块
1. 在 `include/` 和 `src/` 下创建模块目录（如 `new_module`），添加 `.hpp` 和 `.cpp`。
2. 在模块目录中编写 `CMakeLists.txt` 定义库并链接依赖。
3. 在顶层 `CMakeLists.txt` 中添加 `add_subdirectory(src/new_module)`。
4. 若需配置参数，在 `config/` 下新建 YAML 文件，通过 `ConfigManager` 读取。
5. 编写测试程序验证。

## 注意事项
- 输入图像必须为左右拼接格式（宽为单眼两倍）。
- 修改配置后需手动复制 `config/` 到可执行文件目录（或重新编译）。
- 标定结果生成在编译输出目录，如需纳入源码管理请手动复制（程序结束会提示命令）。
- `main.cpp` 实现了生产者-消费者模型和帧率控制，可作为集成参考。
