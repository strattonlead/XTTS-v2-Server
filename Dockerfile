# Blackwell-ready: PyTorch 2.7 + CUDA 12.8 + cuDNN 9
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VOICES_DIR=/app/voices \
    DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1

# Audio- und Runtime-Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Nur exakt die benötigten Dateien kopieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./ 
RUN mkdir -p ${VOICES_DIR}

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
