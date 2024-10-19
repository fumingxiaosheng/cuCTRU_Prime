#!/bin/bash

# 检查参数是否正确
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# 切换到指定目录
cd $1 || exit 1

# 循环遍历目录中的每个文件
for file in *; do
    # 跳过目录
    if [ -d "$file" ]; then
        continue
    fi
    
    # 运行文件
    echo "Running $file ..."
    ./"$file"
    echo "Finished running $file"
done