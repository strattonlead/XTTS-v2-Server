import io
import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf

# Prevent CPML prompt in non-interactive envs
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from TTS.api import TTS

app = FastAPI(title="XTTS-v2 TTS Server", version="1.0.2")

# --- Configuration
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "de")
DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "24000"))

from threading import Lock
_infer_lock = Lock()
_tts: Optional[TTS] = None


def get_tts() -> TTS:
    """Load the TTS model once and cache it globally."""
    global _tts
    if _tts is None:
        tts = TTS(MODEL_NAME)
        tts = tts.to(DEVICE)
        _tts = tts
    return _tts


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}


class SynthesizeJSON(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field(None, description="Language code, e.g. 'de', 'en', 'fr'")
    speaker_wav_url: Optional[str] = Field(None, description="HTTP(s) URL to a short reference WAV for voice cloning")
    sample_rate: Optional[int] = Field(None, description="Target sample rate for WAV output (default 24000)")


@app.post("/synthesize", response_class=StreamingResponse, summary="Synthesize via JSON (optional URL to speaker WAV)")
def synthesize_json(payload: SynthesizeJSON):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    language = (payload.language or DEFAULT_LANGUAGE).strip()
    sr = payload.sample_rate or DEFAULT_SAMPLE_RATE

    speaker_wav_path = None

    # If a URL is provided, fetch into a temp file
    if payload.speaker_wav_url:
        import requests
        try:
            r = requests.get(payload.speaker_wav_url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download speaker_wav_url: {e}")
        fd, speaker_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(speaker_wav_path, "wb") as f:
            f.write(r.content)

    try:
        with _infer_lock:
            tts = get_tts()
            wav = tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(content=buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        if speaker_wav_path and os.path.exists(speaker_wav_path):
            try:
                os.remove(speaker_wav_path)
            except Exception:
                pass


@app.post("/synthesize-multipart", response_class=StreamingResponse, summary="Synthesize with uploaded speaker WAV (multipart/form-data)")
def synthesize_multipart(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None, description="Small reference WAV for voice cloning"),
):
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    lang = (language or DEFAULT_LANGUAGE).strip()
    sr = sample_rate or DEFAULT_SAMPLE_RATE

    tmp_path = None
    if speaker_wav is not None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(speaker_wav.file.read())

    try:
        with _infer_lock:
            tts = get_tts()
            wav = tts.tts(text=text, speaker_wav=tmp_path, language=lang)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(content=buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/")
def root():
    return JSONResponse({
        "name": "XTTS-v2 TTS Server",
        "health": "/health",
        "synthesize_json": {
            "method": "POST",
            "path": "/synthesize",
            "body": {"text": "Hallo Welt", "language": "de",
                     "speaker_wav_url": "https://example.com/ref.wav (optional)", "sample_rate": 24000}
        },
        "synthesize_multipart": {
            "method": "POST",
            "path": "/synthesize-multipart",
            "form": ["text", "language?", "sample_rate?", "speaker_wav? (file)"]
        }
    })
