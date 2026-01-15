# UI-TARS 1.5-7B Deployment via vLLM
# Based on vLLM OpenAI-compatible server image

FROM vllm/vllm-openai:v0.7.3

# Set environment variables
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_ID=ByteDance-Seed/UI-TARS-1.5-7B
ENV PORT=8000
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.90

# Install huggingface_hub for model download
RUN pip install --no-cache-dir huggingface_hub

# Pre-download model weights (speeds up container startup)
# This adds ~15GB to the image but avoids download on every start
RUN huggingface-cli download ${MODEL_ID} --local-dir /models/${MODEL_ID}

EXPOSE ${PORT}

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start vLLM server with OpenAI-compatible API
# Using local model path to avoid re-downloading
CMD python -m vllm.entrypoints.openai.api_server \
    --model /models/${MODEL_ID} \
    --served-model-name ${MODEL_ID} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image":1,"video":0}'
