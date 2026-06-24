"""
KATSconnect TTS server — vLLM + SNAC, running Sunbird's orpheus-3b-tts-multilingual.

Architecture:
  Render backend (kats-server/routes/tts.js) → HF Inference Endpoint (this container)
  → vLLM generates audio tokens → SNAC decoder → WAV bytes → base64 JSON back.

Endpoints:
  GET  /health  — readiness probe for HF IE
  POST /        — inference, accepts {"inputs": str, "parameters": {...}}

Env vars (set by HF IE config):
  HF_TOKEN       — token with read access to the model fork (REQUIRED)
  MODEL_REPO_ID  — override the default fork (optional)
  PORT           — server port (default 8080, HF IE convention)
"""
import os
import io
import sys
import base64
import logging
import time
from typing import Any, Dict, Optional

import torch
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from snac import SNAC
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("orpheus-tts-server")

# Orpheus special token IDs — must match Sunbird's training format.
END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
PAD_TOKEN = 128263
AUDIO_TOKEN_LO = 128266
CODEBOOK_SIZE = 4096
FRAMES_PER_GROUP = 7
AUDIO_TOKEN_HI = AUDIO_TOKEN_LO + FRAMES_PER_GROUP * CODEBOOK_SIZE
SAMPLE_RATE = 24000

DEFAULT_SPEAKER = "salt_eng_0002"
# 4000 tokens ≈ 30s of audio per Sunbird README (1200 → 9-10s, scale linearly).
# Generation time scales too (~35 tok/s on L4 = 114s wall for 4000 tokens
# single-call). For long legal replies, the backend splits into parallel
# segments via vLLM batching, so per-segment cost stays reasonable.
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1

DEFAULT_REPO_ID = "Ojey007/orpheus-3b-tts-multilingual"

# Globals populated by load_model_and_codec at startup.
llm: Optional[LLM] = None
tokenizer = None
snac_model = None
snac_device = "cpu"
ready = False

app = FastAPI(title="KATSconnect TTS")


class TTSRequest(BaseModel):
    inputs: str
    parameters: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def load_model_and_codec():
    """Download model from HF Hub, load vLLM + SNAC. Runs once at boot."""
    global llm, tokenizer, snac_model, snac_device, ready
    t0 = time.time()

    repo_id = os.environ.get("MODEL_REPO_ID", DEFAULT_REPO_ID)
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN env var not set — download will only work for public repos")

    logger.info("Downloading model from %s", repo_id)
    model_path = snapshot_download(
        repo_id=repo_id,
        token=token,
        ignore_patterns=["handler.py", "*.md", ".gitattributes"],
    )
    logger.info("Model downloaded to %s (%.1fs)", model_path, time.time() - t0)

    logger.info("Loading vLLM engine")
    # dtype="float16" works on both T4 (compute 7.5) and L4 (compute 8.9).
    # bfloat16 would be preferable (matches the model's training precision)
    # but it requires compute capability 8.0+, which excludes T4. Quality
    # difference on TTS is imperceptible in practice.
    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Loading SNAC 24kHz codec")
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(snac_device)
    snac_model.eval()

    ready = True
    logger.info("Server ready in %.1fs", time.time() - t0)


@app.get("/health")
async def health():
    if not ready:
        raise HTTPException(503, "model still loading")
    return {"status": "ok"}


@app.post("/")
async def synthesize(req: TTSRequest):
    if not ready:
        raise HTTPException(503, "model still loading")

    text = (req.inputs or "").strip()
    if not text:
        raise HTTPException(400, "inputs must be a non-empty string")

    params = req.parameters or {}
    speaker_id = params.get("speaker_id", DEFAULT_SPEAKER)
    max_tokens = int(params.get("max_new_tokens", DEFAULT_MAX_TOKENS))
    temperature = float(params.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(params.get("top_p", DEFAULT_TOP_P))
    repetition_penalty = float(params.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
    seed = params.get("seed")

    t0 = time.time()
    try:
        tagged = f"{speaker_id}: {text}"
        text_ids = tokenizer.encode(tagged, add_special_tokens=True)
        prompt_token_ids = [START_OF_HUMAN] + text_ids + [END_OF_TEXT, END_OF_HUMAN]

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            stop_token_ids=[END_OF_SPEECH],
            skip_special_tokens=False,
            seed=int(seed) if seed is not None else None,
        )

        outputs = llm.generate([{"prompt_token_ids": prompt_token_ids}], sampling)
        generated_token_ids = list(outputs[0].outputs[0].token_ids)
        gen_time = time.time() - t0

        audio_bytes = _decode_to_wav(generated_token_ids)
        total_time = time.time() - t0

        logger.info(
            "synthesize speaker=%s chars=%d tokens=%d gen=%.2fs total=%.2fs",
            speaker_id, len(text), len(generated_token_ids), gen_time, total_time,
        )

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": SAMPLE_RATE,
            "format": "wav",
            "speaker_id": speaker_id,
            "input_chars": len(text),
            "generated_tokens": len(generated_token_ids),
            "generation_ms": int(gen_time * 1000),
            "total_ms": int(total_time * 1000),
        }
    except Exception as e:
        logger.exception("inference failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _decode_to_wav(generated_token_ids):
    """Convert generated audio tokens to WAV bytes via SNAC."""
    ids = torch.tensor(generated_token_ids, dtype=torch.int64)

    # Crop to everything AFTER the last START_OF_SPEECH marker the model emitted
    sos_pos = (ids == START_OF_SPEECH).nonzero(as_tuple=True)[0]
    if len(sos_pos) > 0:
        ids = ids[sos_pos[-1].item() + 1:]

    # Keep only tokens inside the audio codebook range
    audio_ids = ids[(ids >= AUDIO_TOKEN_LO) & (ids < AUDIO_TOKEN_HI)]
    n = (audio_ids.size(0) // FRAMES_PER_GROUP) * FRAMES_PER_GROUP
    if n == 0:
        raise ValueError(f"no audio tokens in output (got {len(generated_token_ids)} generated)")

    code_list = [int(t.item()) - AUDIO_TOKEN_LO for t in audio_ids[:n]]

    # Redistribute the flat 7-token-per-frame stream into SNAC's 3 codebook layers.
    # Layout per frame: position 0 → layer 1 (coarse), positions 1+4 → layer 2,
    # positions 2/3/5/6 → layer 3 (fine). Per-position offsets subtracted to
    # bring each code into [0, 4096).
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(len(code_list) // FRAMES_PER_GROUP):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - 2 * 4096)
        layer_3.append(code_list[7 * i + 3] - 3 * 4096)
        layer_2.append(code_list[7 * i + 4] - 4 * 4096)
        layer_3.append(code_list[7 * i + 5] - 5 * 4096)
        layer_3.append(code_list[7 * i + 6] - 6 * 4096)

    clamp = lambda vals: [max(0, min(4095, v)) for v in vals]
    codes = [
        torch.tensor(clamp(layer_1), dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(clamp(layer_2), dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(clamp(layer_3), dtype=torch.int32, device=snac_device).unsqueeze(0),
    ]

    with torch.inference_mode():
        waveform = snac_model.decode(codes)
    audio = waveform.detach().squeeze().cpu().numpy().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
