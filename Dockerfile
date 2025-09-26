FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VOICES_DIR=/app/voices \
    COQUI_TOS_AGREED=1

# System deps (Audio + optional phonemizer-deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
RUN mkdir -p ${VOICES_DIR}

# (Optional) Modell schon beim Build cachen – beschleunigt den 1. Request
RUN python - <<'PY'
from TTS.api import TTS
TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("XTTS cached.")
PY

EXPOSE 8000
ENV DEVICE=cuda
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]