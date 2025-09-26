# XTTS-v2 TTS Server (CUDA)

A FastAPI-based server around [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) with CUDA support.
This version includes **speaker enrollment**, **JSON and multipart synthesis**, **persistent storage**, and **OpenAPI docs**.

---

## Features

- **Health Check**: `/health`
- **Speaker Management**
  - `POST /speakers/register` → Register (enroll) a new speaker from WAV upload or URL
  - `GET /speakers` → List all registered speakers
  - `DELETE /speakers/{speaker_id}` → Remove a registered speaker
- **Synthesis**
  - `POST /synthesize` → JSON request
  - `POST /synthesize-multipart` → Multipart/form-data request
- **Built-in Speakers**: use `speaker_0`, `speaker_1`, etc. (from the model)
- **Voice Cloning**: provide your own short WAV reference
- **Persistent storage** for model cache and registered speakers (`/root/.local/share/tts`, `/data`)
- **Interactive OpenAPI UI** at `/docs`
- **Raw OpenAPI spec** at `/openapi.json`

---

## Environment Variables

| Variable             | Default Value                                            | Description |
|----------------------|----------------------------------------------------------|-------------|
| `COQUI_TOS_AGREED`   | `1`                                                      | Must be set to confirm Coqui's CPML license (or commercial license). |
| `TTS_MODEL_NAME`     | `tts_models/multilingual/multi-dataset/xtts_v2`          | Hugging Face model name. |
| `DEVICE`             | `cuda` if GPU available, otherwise `cpu`                 | Compute device. |
| `DEFAULT_LANGUAGE`   | `de`                                                     | Default language. |
| `DEFAULT_SAMPLE_RATE`| `24000`                                                  | Default audio sample rate. |
| `DATA_DIR`           | `/data`                                                  | Persistent storage path for registered speakers. |

---

## Run the Container

### With GPU (recommended)
```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e COQUI_TOS_AGREED=1 \
  -e DEVICE=cuda \
  -v $(pwd)/coqui_cache:/root/.local/share/tts \
  -v $(pwd)/data:/data \
  createiflabs/xtts-vs-server-cuda:latest
