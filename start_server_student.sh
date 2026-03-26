#!/bin/bash

MODEL_PATH="Qwen/Qwen3-8B"     # 模型名或本地路径
HOST="0.0.0.0"                          # 绑定所有网络接口
PORT=23334                              # 推荐使用 20000~29999 之间的端口
TP=8                                    # 根据 GPU 数量调整
GPU_MEM_UTILIZATION=0.3
MAX_MODEL_LEN=40960
MODEL_NAME="Qwen/Qwen3-8B"


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


