#!/bin/bash
# CHUSEI 3D摄像头初始化脚本
DEVICE=$1
if [ -z "$DEVICE" ]; then
    echo "错误: 未指定设备路径"
    exit 1
fi
echo "正在为设备 $DEVICE 执行 CHUSEI 初始化命令..."
commands=(
    "uvcdynctrl -d $DEVICE -S 6:8  '(LE)0x50ff'"
    "uvcdynctrl -d $DEVICE -S 6:15 '(LE)0x00f6'"
    "uvcdynctrl -d $DEVICE -S 6:8  '(LE)0x2500'"
    "uvcdynctrl -d $DEVICE -S 6:8  '(LE)0x5ffe'"
    "uvcdynctrl -d $DEVICE -S 6:15 '(LE)0x0003'"
    "uvcdynctrl -d $DEVICE -S 6:15 '(LE)0x0002'"
    "uvcdynctrl -d $DEVICE -S 6:15 '(LE)0x0012'"
    "uvcdynctrl -d $DEVICE -S 6:15 '(LE)0x0004'"
    "uvcdynctrl -d $DEVICE -S 6:8  '(LE)0x76c3'"
    "uvcdynctrl -d $DEVICE -S 6:10 '(LE)0x0400'"
)
for cmd in "${commands[@]}"; do
    echo "执行: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "警告: 命令执行失败: $cmd"
    fi
    sleep 0.1
done
echo "CHUSEI 初始化命令执行完毕"
exit 0
