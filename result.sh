#!/bin/bash

# 初始化变量，用于存储总和和计数
total=0
count=0
filename="$1"

echo "$filename"

# 循环运行 build/test653 一百次
for ((i = 1; i <= 5000; i++)); do
    # 运行 build/test653 并将结果保存到变量 result 中
    result=$(./$filename)
    
    # 输出当前运行的结果
    #echo "Run $i: $result"
    
    # 将结果添加到总和中
    total=$(echo "$total + $result" | bc)
    
    # 增加计数器
    count=$((count + 1))
done

# 计算平均值
average=$(echo "scale=6; $total / $count" | bc)

# 输出平均值
echo "Average: $average"
