# XTTS-v2 FastAPI + Uvicorn (CUDA) – Dockerized TTS Server

This project provides an HTTP server for [coqui/XTTS-v2].

- **Text → Audio** (WAV)
- **Language** controllable via parameter
- **Speaker voice cloning** via reference WAV (upload or URL)
- Runs with **CUDA** (GPU) if available

---

## Files

- `app/main.py` – FastAPI/Uvicorn server
- `requirements.txt` – Python dependencies
- `Dockerfile` – Build for CUDA runtime
- `.env.example` – Example configuration

---

## Docker Build & Run

### Build
```bash
docker build -t xtts-v2-server:cuda .
```

### Run with GPU
```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2 \
  -e DEVICE=cuda \
  -e DEFAULT_LANGUAGE=en \
  -e DEFAULT_SAMPLE_RATE=24000 \
  xtts-v2-server:cuda
```

### Run with `.env` file
```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  xtts-v2-server:cuda
```

---

## Environment Variables

| Variable            | Default Value                                           | Description                                                                 |
|---------------------|---------------------------------------------------------|-----------------------------------------------------------------------------|
| `TTS_MODEL_NAME`    | `tts_models/multilingual/multi-dataset/xtts_v2`         | HuggingFace/Coqui model ID                                                  |
| `DEVICE`            | `cuda` (if GPU available, otherwise `cpu`)              | Device for inference                                                        |
| `DEFAULT_LANGUAGE`  | `en`                                                    | Default language for synthesis                                              |
| `DEFAULT_SAMPLE_RATE` | `24000`                                               | Sample rate for output WAV                                                  |

---

## API Endpoints

### Health
```bash
curl -s http://localhost:8000/health | jq
```
Response:
```json
{
  "status": "ok",
  "device": "cuda",
  "model": "tts_models/multilingual/multi-dataset/xtts_v2"
}
```

### JSON Endpoint (`/synthesize`)

```bash
curl -X POST http://localhost:8000/synthesize \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello! This is a test.",
    "language": "en",
    "speaker_wav_url": "https://example.com/my-voice.wav",
    "sample_rate": 24000
  }' \
  --output out.wav
```

- `text`: Required, text to synthesize
- `language`: Language code (`en`, `de`, `fr`, ...)
- `speaker_wav_url`: URL to a reference WAV (optional)
- `sample_rate`: Desired sample rate (optional)

### Multipart Endpoint (`/synthesize-multipart`)

```bash
curl -X POST http://localhost:8000/synthesize-multipart \
  -F text='Good morning!' \
  -F language='en' \
  -F sample_rate=24000 \
  -F speaker_wav=@/path/to/reference.wav \
  --output out.wav
```

- `speaker_wav` can be uploaded as a file

---

## Project Structure

```
.
├── app/
│   └── main.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Notes

- **GPU**: Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **Speaker voice**: Use 3–10 seconds of clean mono WAV as a reference.
- **Scaling**: Run multiple containers and use load balancing for higher throughput.
- **Security**: Add authentication and request limits if exposed publicly.

---

Enjoy testing 🚀