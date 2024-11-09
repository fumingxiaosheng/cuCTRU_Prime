#!/bin/bash

# 设置脚本名称和描述
SCRIPT_NAME="./build/test1277"
TIMES=1000

# 检查命令是否存在
if ! command -v $SCRIPT_NAME &> /dev/null; then
    echo "Error: The script '$SCRIPT_NAME' does not exist or is not executable."
    exit 1
fi

# 运行命令指定次数
for ((i=1; i<=TIMES; i++)); do
    echo "Running $SCRIPT_NAME (Iteration $i/$TIMES)"
    $SCRIPT_NAME
done

echo "Finished running $SCRIPT_NAME $TIMES times."