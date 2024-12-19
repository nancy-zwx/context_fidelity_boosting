#!/bin/bash

# 寻找并终止所有运行 run_anchun.py 的进程
echo "正在查找并终止占用GPU的进程..."

# 找到所有运行 run_anchun.py 的python进程
PIDS=$(ps aux | grep "run_anchun.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有找到占用GPU的进程"
else
    echo "找到以下进程："
    for PID in $PIDS; do
        echo "进程ID: $PID"
        echo "进程信息："
        ps -p $PID -o pid,ppid,user,%cpu,%mem,cmd
    done

    echo -n "是否终止这些进程？(y/n): "
    read answer
    if [ "$answer" = "y" ]; then
        for PID in $PIDS; do
            echo "正在终止进程 $PID"
            kill -9 $PID
        done
        echo "所有进程已终止"
    else
        echo "操作已取消"
    fi
fi

# 验证GPU状态
echo -e "\nGPU状态："
nvidia-smi

# sh /apdcephfs_cq10/share_1567347/share_info/wendyzhang/gpu/kill.sh