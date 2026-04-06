#!/bin/bash

# 项目根目录（脚本所在目录）
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 需要处理的目录（相对路径，可自行增删）
SOURCE_DIRS=("include" "src")

# 版权声明模板（根据你的项目信息修改）
COPYRIGHT_NOTICE='// Copyright (C) 2026 C01-JNU
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

'

# 判断文件是否已有版权声明（简单检查是否包含 "Copyright" 或 "SPDX-License-Identifier"）
has_copyright() {
    grep -q -E "(Copyright|SPDX-License-Identifier)" "$1"
}

# 添加头部
add_header() {
    local file="$1"
    local temp_file="${file}.tmp"
    
    # 将版权声明和原文件内容合并
    echo "$COPYRIGHT_NOTICE" > "$temp_file"
    cat "$file" >> "$temp_file"
    
    # 替换原文件
    mv "$temp_file" "$file"
    echo "  Added header to $file"
}

# 遍历所有指定目录
for dir in "${SOURCE_DIRS[@]}"; do
    full_path="${PROJECT_ROOT}/${dir}"
    if [ ! -d "$full_path" ]; then
        echo "Warning: Directory $full_path does not exist, skipping."
        continue
    fi
    
    find "$full_path" -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.c" \) | while read -r file; do
        if has_copyright "$file"; then
            echo "Skip (already has copyright): $file"
        else
            add_header "$file"
        fi
    done
done

echo "Done."
