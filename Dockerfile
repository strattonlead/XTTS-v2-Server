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

# Workdir
WORKDIR /app

RUN mkdir -p ${VOICES_DIR}

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py ./

# Expose
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]