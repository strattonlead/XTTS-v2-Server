from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from TTS.api import TTS
import torch
import os, io, glob, shutil
import soundfile as sf  # für Validierung/Laden
import numpy as np

app = FastAPI(title="XTTS Server", version="1.0.0")

# ---- Config ----
# XTTS v2 (Coqui) – mehrsprachig, Voice-Cloning via speaker_wav(s)
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
VOICES_DIR = os.getenv("VOICES_DIR", "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Device Wahl (ENV DEVICE=cpu/cuda überschreibt Auto)
device = os.getenv("DEVICE")
if device not in {"cpu", "cuda"}:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Map ISO3 -> ISO2 (falls du alte Codes nutzen willst)
ISO3_TO_ISO2 = {"deu": "de", "eng": "en"}

# ---- Load model once ----
tts = TTS(MODEL_NAME).to(device)

class TTSRequest(BaseModel):
    text: str
    lang: str  # z.B. "de" / "en" (oder "deu"/"eng" -> wird gemappt)
    voice: Optional[str] = None       # Name eines Voice-Ordners unter ./voices
    speed: Optional[float] = 1.0      # 0.5..1.5, XTTS unterstützt speed
    seed: Optional[int] = None        # für Reproduzierbarkeit (optional)
    split_sentences: Optional[bool] = True

def _lang_norm(lang: str) -> str:
    l = lang.lower()
    return ISO3_TO_ISO2.get(l, l)

def _collect_voice_samples(voice_name: Optional[str]) -> Optional[List[str]]:
    if not voice_name:
        return None
    vdir = os.path.join(VOICES_DIR, voice_name)
    if not os.path.isdir(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")
    files = sorted(glob.glob(os.path.join(vdir, "*.wav")))
    if not files:
        raise HTTPException(status_code=400, detail=f"No .wav files in voice '{voice_name}'.")
    return files

@app.get("/health")
def health():
    cap = torch.cuda.get_device_capability(0) if (device == "cuda" and torch.cuda.is_available()) else None
    return {
        "status": "ok",
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "gpu_capability": cap,
        "model": MODEL_NAME,
    }

@app.get("/voices")
def list_voices():
    voices = []
    for d in sorted(os.listdir(VOICES_DIR)):
        p = os.path.join(VOICES_DIR, d)
        if os.path.isdir(p):
            n = len(glob.glob(os.path.join(p, "*.wav")))
            voices.append({"name": d, "num_samples": n})
    return {"voices": voices}

@app.post("/voices/{name}")
async def add_voice(name: str, files: List[UploadFile] = File(...)):
    # Lege/Leere Zielordner an
    vdir = os.path.join(VOICES_DIR, name)
    os.makedirs(vdir, exist_ok=True)

    saved = 0
    for i, f in enumerate(files, start=1):
        if not f.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail=f"Only .wav supported (got {f.filename}).")
        data = await f.read()
        # Validierung & ggf. Re-Save (stellt sicher, dass es wirklich PCM WAV ist)
        try:
            buf = io.BytesIO(data)
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid WAV ({f.filename}): {e}")
        out = os.path.join(vdir, f"{i:02d}.wav")
        sf.write(out, audio, sr)
        saved += 1

    return {"ok": True, "voice": name, "saved": saved}

@app.delete("/voices/{name}")
def delete_voice(name: str):
    vdir = os.path.join(VOICES_DIR, name)
    if not os.path.isdir(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")
    shutil.rmtree(vdir)
    return {"ok": True, "deleted": name}

@app.post("/tts")
def synthesize(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    lang = _lang_norm(req.lang)
    if req.seed is not None:
        torch.manual_seed(int(req.seed))
        np.random.seed(int(req.seed))

    speaker_wavs = _collect_voice_samples(req.voice)  # optional list of .wav paths

    speaker_arg = speaker_wavs if speaker_wavs and len(speaker_wavs) > 1 else (speaker_wavs[0] if speaker_wavs else None)

    wav = tts.tts(
        text=req.text.strip(),
        language=lang,
        speaker_wav=speaker_arg,
        speed=req.speed if req.speed else 1.0,
        split_sentences=bool(req.split_sentences),
    )
    # Rückgabe kann numpy array (mono) sein; sr aus tts.synthesizer.output_sample_rate
    sr = getattr(tts.synthesizer, "output_sample_rate", 24000)

    # In WAV serialize (16-bit PCM)
    audio = np.asarray(wav, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm16, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)

    # Datei-Download
    base = f"xtts-{lang}"
    if req.voice:
        base += f"-{req.voice}"
    filename = f"{base}.wav"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)
