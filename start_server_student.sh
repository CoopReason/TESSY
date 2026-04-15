#!/bin/bash

MODEL_PATH="Qwen/Qwen3-8B"     # Path or HuggingFace name of the model
HOST="0.0.0.0"                # Bind to all network interfaces (accessible externally)
PORT=23334                    # Port to serve the API (recommended: 20000–29999 range)
TP=8                          # Tensor parallel size (set based on number of GPUs)
GPU_MEM_UTILIZATION=0.3       # Fraction of GPU memory to use (0~1)
MAX_MODEL_LEN=40960           # Maximum context length supported by the model
MODEL_NAME="Qwen/Qwen3-8B"    # Name exposed to clients via OpenAI API


# ------------------ Launch vLLM API Server ------------------
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTILIZATION \
    --served-model-name $MODEL_NAME \
    --trust-remote-code        
