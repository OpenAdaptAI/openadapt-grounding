# UI-TARS 1.5-7B Deployment via vLLM
# Based on vLLM OpenAI-compatible server image

FROM vllm/vllm-openai:v0.7.3

# Model will be downloaded at runtime to /root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_NAME=ByteDance-Seed/UI-TARS-1.5-7B

EXPOSE 8000

# Health check - start period allows for model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start vLLM server - model downloaded at first start
# Note: vllm-openai image has ENTRYPOINT that runs api_server, so CMD is just args
CMD ["--model", "ByteDance-Seed/UI-TARS-1.5-7B", \
     "--served-model-name", "ByteDance-Seed/UI-TARS-1.5-7B", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--trust-remote-code", \
     "--limit-mm-per-prompt", "image=1"]
