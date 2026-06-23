# Extends vLLM's official image so we don't fight package version conflicts.
# vLLM + torch + CUDA + transformers are already pre-installed and known-good.
# We just add SNAC + soundfile (audio codec/IO) and our FastAPI service.
FROM vllm/vllm-openai:latest

WORKDIR /workspace

# Install only the additional deps we need on top of the vLLM base.
# Everything else (vllm, torch, transformers, numpy, fastapi) comes pre-installed.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Our inference service (replaces vLLM's default OpenAI-compatible server).
COPY server.py /workspace/server.py

# Container expects HF_TOKEN env var (set by HF IE) to pull the model fork.
# MODEL_REPO_ID env var overrides the default fork — leave unset to use the one
# baked into server.py.

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# HF IE pings GET /health for readiness and routes inference to POST /
EXPOSE 8080

# Override the vLLM base image's ENTRYPOINT (which would run vLLM's OpenAI server).
ENTRYPOINT []
CMD ["python3", "/workspace/server.py"]
