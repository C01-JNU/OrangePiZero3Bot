#!/bin/bash
# 立体图像分割工具 (Shell版本)
# 使用ImageMagick进行图像分割

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}立体图像分割工具${NC}"
    echo "将拼接的立体图像分割为左右眼图像，按标定程序命名格式保存"
    echo ""
    echo "用法: $0 <输入目录> <输出目录> [起始编号]"
    echo ""
    echo "参数:"
    echo "  <输入目录>：包含拼接图像的目录"
    echo "  <输出目录>：输出左右眼图像的目录"
    echo "  [起始编号]：可选，起始编号（默认：0）"
    echo ""
    echo "示例:"
    echo "  $0 images/test images/calibration 0"
    echo ""
    echo "要求:"
    echo "  - 系统已安装ImageMagick (convert命令)"
    echo "  - 图像格式支持: jpg, jpeg, png"
    echo "  - 图像宽度为偶数"
    echo ""
    echo "输出格式:"
    echo "  000_left.jpg, 000_right.jpg, 001_left.jpg, 001_right.jpg, ..."
}

# 检查参数
if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
START_INDEX="${3:-0}"

# 检查ImageMagick是否安装
if ! command -v convert &> /dev/null; then
    echo -e "${RED}错误: ImageMagick未安装${NC}"
    echo "请安装: sudo apt-get install imagemagick"
    exit 1
fi

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 清空输出目录中的旧文件
rm -f "$OUTPUT_DIR"/*_left.jpg "$OUTPUT_DIR"/*_right.jpg

# 获取图像文件列表
IMAGE_FILES=()
while IFS= read -r -d $'\0' file; do
    IMAGE_FILES+=("$file")
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print0 | sort -z)

# 检查是否有图像文件
if [ ${#IMAGE_FILES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 输入目录中没有图像文件${NC}"
    echo "支持的格式: .jpg, .jpeg, .png"
    exit 1
fi

echo -e "${GREEN}开始处理图像分割...${NC}"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "起始编号: $START_INDEX"
echo "找到 ${#IMAGE_FILES[@]} 张图像"
echo "----------------------------------------"

# 处理每张图像
SUCCESS_COUNT=0
ERROR_COUNT=0
INDEX=$START_INDEX

for i in "${!IMAGE_FILES[@]}"; do
    IMAGE_PATH="${IMAGE_FILES[$i]}"
    FILENAME=$(basename "$IMAGE_PATH")
    
    # 计算进度
    PROGRESS=$(( (i + 1) * 100 / ${#IMAGE_FILES[@]} ))
    
    echo -e "${BLUE}[$((i+1))/${#IMAGE_FILES[@]}] (${PROGRESS}%) 处理: $FILENAME${NC}"
    
    # 获取图像宽度
    WIDTH=$(identify -format "%w" "$IMAGE_PATH" 2>/dev/null)
    
    if [ -z "$WIDTH" ]; then
        echo -e "  ${RED}错误: 无法读取图像尺寸${NC}"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi
    
    # 检查宽度是否为偶数
    if [ $((WIDTH % 2)) -ne 0 ]; then
        echo -e "  ${YELLOW}警告: 图像宽度为奇数 ($WIDTH)，将调整宽度为偶数${NC}"
        WIDTH=$((WIDTH - 1))
    fi
    
    # 计算单眼宽度
    SINGLE_WIDTH=$((WIDTH / 2))
    
    # 格式化编号（3位数字）
    INDEX_STR=$(printf "%03d" $INDEX)
    
    # 左眼图像路径
    LEFT_PATH="$OUTPUT_DIR/${INDEX_STR}_left.jpg"
    # 右眼图像路径
    RIGHT_PATH="$OUTPUT_DIR/${INDEX_STR}_right.jpg"
    
    # 分割图像
    if convert "$IMAGE_PATH" -crop ${SINGLE_WIDTH}x+0+0 +repage "$LEFT_PATH" 2>/dev/null && \
       convert "$IMAGE_PATH" -crop ${SINGLE_WIDTH}x+${SINGLE_WIDTH}+0 +repage "$RIGHT_PATH" 2>/dev/null; then
        
        # 获取输出图像尺寸
        LEFT_SIZE=$(identify -format "%wx%h" "$LEFT_PATH" 2>/dev/null)
        RIGHT_SIZE=$(identify -format "%wx%h" "$RIGHT_PATH" 2>/dev/null)
        
        echo -e "  ${GREEN}[$INDEX_STR] 左眼: $LEFT_PATH ($LEFT_SIZE)${NC}"
        echo -e "  ${GREEN}[$INDEX_STR] 右眼: $RIGHT_PATH ($RIGHT_SIZE)${NC}"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        INDEX=$((INDEX + 1))
    else
        echo -e "  ${RED}错误: 图像分割失败${NC}"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
done

echo "----------------------------------------"
echo -e "${GREEN}处理完成!${NC}"
echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "失败: ${RED}$ERROR_COUNT${NC}"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}输出图像保存在: $OUTPUT_DIR${NC}"
    echo "标定程序用法:"
    echo -e "  1. 将 ${YELLOW}$OUTPUT_DIR${NC} 目录中的图像用于标定"
    echo -e "  2. 运行标定程序: ${YELLOW}./stereo_calibrator${NC}"
    echo ""
    echo -e "${YELLOW}标定程序要求:${NC}"
    echo "  - 图像命名格式: *_left.jpg, *_right.jpg"
    echo "  - 图像尺寸一致"
    echo "  - 每对图像对应相同场景"
fi

if [ $ERROR_COUNT -gt 0 ]; then
    exit 1
fi
