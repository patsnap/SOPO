#!/bin/bash

# 设置日志文件名（包含时间戳）
LOG_FILE="logs/train_refinder_$(date +%Y%m%d_%H%M%S).log"

# 将所有输出（stdout 和 stderr）同时显示在终端和保存到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

export MASTER_IP=10.148.0.108
export MASTER_PORT=23456
ray stop
echo "########### ray start ###########"
ray start --head --num-gpus 8 --node-ip-address=$MASTER_IP --port=$MASTER_PORT --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265
sleep 10s
ray status
echo "########### job submit ###########"
python3 -u -m areal.launcher.ray examples/havor/train.py \
    --config examples/havor/config_capo.yaml
