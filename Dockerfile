# CUDA + PyTorch base (includes torch w/ CUDA). Choose a tag matching your driver.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime


# Avoid interactive tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1
ENV PYTHONWARNINGS="ignore:You are using `torch.load`:FutureWarning"

# System deps for soundfile/libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Optional: set huggingface + coqui caches to persistable locations
ENV HF_HOME=/root/.cache/huggingface \
    COQUI_TTS_CACHE=/root/.local/share/tts

# Workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app

# Env defaults
ENV TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2 \
    DEVICE=cuda \
    DEFAULT_LANGUAGE=de \
    DEFAULT_SAMPLE_RATE=24000 \
    PYTHONUNBUFFERED=1

# Expose
EXPOSE 8000

# Download model at build time (optional, speeds up first run). Comment out if you prefer lazy load at runtime.
# RUN python -c "from TTS.api import TTS; TTS('${TTS_MODEL_NAME}').to('cpu')"

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
