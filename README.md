# katsconnect-tts-server

Docker container that runs Sunbird's `orpheus-3b-tts-multilingual` (Ugandan-accented
English + Luganda + 18 other African languages) via vLLM, serving HTTP requests
for KATSconnect's Resty assistant.

Built on `vllm/vllm-openai:latest` so vLLM + CUDA + torch versions are
known-compatible. Adds SNAC (the 24kHz audio codec the model emits codes into) +
a small FastAPI service that wraps the inference pipeline.

## Architecture

```
KATSconnect frontend (Resty modal)
  ↓ POST /api/tts/speak  (kats-server route)
Render backend (routes/tts.js)
  ↓ POST /  (Bearer KATS_TTS_TOKEN)
HF Inference Endpoint (this container)
  ├─ vLLM generates audio tokens from text + speaker_id
  ├─ SNAC decoder converts tokens → 24kHz PCM
  └─ base64 WAV returned as JSON
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness probe (returns 200 when model is loaded) |
| POST | `/` | Inference. Body: `{"inputs": str, "parameters": {"speaker_id": "salt_eng_0002", ...}}` |

Response shape:
```json
{
  "audio_base64": "...",
  "sample_rate": 24000,
  "format": "wav",
  "speaker_id": "salt_eng_0002",
  "input_chars": 42,
  "generated_tokens": 1837,
  "generation_ms": 14250,
  "total_ms": 14380
}
```

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | yes | HF fine-grained token with read access to the model fork |
| `MODEL_REPO_ID` | no | Override the default `Ojey007/orpheus-3b-tts-multilingual` |
| `PORT` | no | HTTP port (default `8080`) |

## How CI/CD works

Pushing to `main` triggers `.github/workflows/docker-build.yml`, which:
1. Builds the image with Docker Buildx (cached layers via GHA cache)
2. Pushes to `ghcr.io/ojey/katsconnect-tts-server:latest` (and the commit SHA as a tag)

First successful build creates the GHCR package as **private** by default. To let
HF IE pull it without auth, make it public:

1. https://github.com/users/Ojey/packages/container/katsconnect-tts-server/settings
2. Scroll to "Danger Zone" → "Change visibility" → Public

## HF IE configuration

When creating/updating the Inference Endpoint:

| Field | Value |
|---|---|
| Inference Engine | **Custom** |
| Container URL | `ghcr.io/ojey/katsconnect-tts-server:latest` |
| Container port | `8080` |
| Health route | `/health` |
| Environment variable: `HF_TOKEN` | (your read-scoped HF token) |

## Voice catalog

Primary voices Resty uses:
- `salt_eng_0002` — female Ugandan English (default)
- `salt_lug_0001` — female Luganda

Underlying model supports ~50 more speakers across Acholi, Ateso, Runyankole,
Hausa, Yoruba, Swahili, etc. — switch by passing a different `speaker_id`.
