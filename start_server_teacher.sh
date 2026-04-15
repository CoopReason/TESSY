#!/bin/bash

# ------------------ Model & Service Configuration ------------------
MODEL_PATH="openai/gpt-oss-120b"   # Model name (HuggingFace) or local path
HOST="0.0.0.0"                    # Bind to all network interfaces (allows external access)
PORT=23333                        # Port for the API server (recommended: 20000–29999)
TP=8                              # Tensor parallel size (should match number of GPUs)
GPU_MEM_UTILIZATION=0.65          # Fraction of GPU memory to allocate (range: 0~1)
MAX_MODEL_LEN=40960               # Maximum supported context length (tokens)
MODEL_NAME="openai/gpt-oss-120b"  # Model name exposed via OpenAI-compatible API


# ------------------ Launch Service ------------------
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTILIZATION \
    --served-model-name $MODEL_NAME \
    --trust-remote-code  
