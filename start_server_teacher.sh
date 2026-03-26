#!/bin/bash

# ------------------ 模型与服务配置 ------------------
MODEL_PATH="openai/gpt-oss-120b"     # 模型名或本地路径
HOST="0.0.0.0"                          # 绑定所有网络接口
PORT=23333                              # 推荐使用 20000~29999 之间的端口
TP=8                                    # 根据 GPU 数量调整
GPU_MEM_UTILIZATION=0.65
MAX_MODEL_LEN=40960
MODEL_NAME="openai/gpt-oss-120b"


# ------------------ 启动服务 ------------------
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTILIZATION \
    --served-model-name $MODEL_NAME \
    --trust-remote-code


