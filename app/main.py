import io
import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path as FPath, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf

# Prevent CPML prompt in non-interactive envs (Coqui CPML / non-commercial or commercial license)
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Optional: point caches to persistent locations if you mount volumes
# os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
# os.environ.setdefault("COQUI_TTS_CACHE", "/root/.local/share/tts")

from TTS.api import TTS

app = FastAPI(
    title="XTTS TTS Server",
    version="2.2.0",
    description="XTTS TTS with speaker enrollment (persistent), per-request model override, model pull, CUDA-ready."
)

# --- Configuration
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "de")
DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "24000"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()  # persistent volume for our app state
SPEAKERS_DIR = DATA_DIR / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

from threading import Lock
_infer_lock = Lock()

# Cache mehrere Modelle/Devices: {(model_name, device): TTS()}
_tts_cache: Dict[Tuple[str, str], TTS] = {}


def get_tts(model_name: Optional[str] = None, device: Optional[str] = None) -> TTS:
    """
    Load the TTS model once and cache it (keyed by model_name + device).
    If model_name/device not given, use defaults from env.
    """
    m = (model_name or MODEL_NAME).strip()
    d = (device or DEVICE).strip()
    key = (m, d)
    tts = _tts_cache.get(key)
    if tts is None:
        tts = TTS(m).to(d)
        _tts_cache[key] = tts
    return tts


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _speaker_path(speaker_id: str) -> Path:
    return SPEAKERS_DIR / f"{speaker_id}.wav"


def _speaker_meta_path(speaker_id: str) -> Path:
    return SPEAKERS_DIR / f"{speaker_id}.json"


def _save_bytes_to(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _register_wav(speaker_id: str, wav_bytes: bytes, meta: Dict):
    wav_path = _speaker_path(speaker_id)
    _save_bytes_to(wav_path, wav_bytes)
    meta_path = _speaker_meta_path(speaker_id)
    meta = {**meta, "speaker_id": speaker_id, "wav_path": str(wav_path)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _load_registered_speakers() -> List[str]:
    return sorted([p.stem for p in SPEAKERS_DIR.glob("*.wav")])


def _ensure_registered(speaker_id: str) -> Path:
    wav_path = _speaker_path(speaker_id)
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown speaker_id '{speaker_id}'. Register first.")
    return wav_path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_default": MODEL_NAME,
        "data_dir": str(DATA_DIR),
        "registered_speakers": _load_registered_speakers(),
    }


# =========================
# Models API (Pull/List/Evict)
# =========================

class PullModelJSON(BaseModel):
    model_name: str = Field(..., description="Hugging Face / Coqui TTS model name, e.g. tts_models/multilingual/multi-dataset/xtts_v2")
    device: Optional[str] = Field(None, description="Override device for warm-load (default uses server DEVICE)")


@app.get("/models", summary="List in-memory loaded models and defaults")
def list_models():
    loaded = [{"model_name": k[0], "device": k[1]} for k in _tts_cache.keys()]
    return {
        "default_model": MODEL_NAME,
        "default_device": DEVICE,
        "loaded": loaded,
        "data_dir": str(DATA_DIR),
    }


@app.post("/models/pull", summary="Download & warm-load a model into cache")
def pull_model(payload: PullModelJSON):
    """
    Forces a model download (if needed) and preloads it into memory on a chosen device.
    It's idempotent: if already cached in-memory, it reuses it.
    """
    model_name = payload.model_name.strip()
    device = (payload.device or DEVICE).strip()
    try:
        with _infer_lock:
            tts = get_tts(model_name=model_name, device=device)
            # Touch a cheap attribute to ensure it's fully ready (and force download during build)
            _ = getattr(tts, "speakers", None)
        return {"ok": True, "model_name": model_name, "device": device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pull model '{model_name}' on device '{device}': {e}")


class EvictModelJSON(BaseModel):
    model_name: str = Field(..., description="Model name to evict from in-memory cache")
    device: Optional[str] = Field(None, description="Device to target; default = server DEVICE")


@app.delete("/models/evict", summary="Evict a model from in-memory cache (files on disk remain)")
def evict_model(payload: EvictModelJSON):
    m = payload.model_name.strip()
    d = (payload.device or DEVICE).strip()
    key = (m, d)
    existed = key in _tts_cache
    # Best effort: close/del the instance
    if existed:
        try:
            _tts_cache.pop(key, None)
        except Exception:
            pass
    return {"ok": True, "evicted": existed, "model_name": m, "device": d}


# =========================
# Speaker Enrollment API
# =========================

class RegisterSpeakerJSON(BaseModel):
    speaker_id: Optional[str] = Field(None, description="Desired ID; if omitted, generated from source.")
    wav_url: Optional[str] = Field(None, description="HTTP(s) URL to reference WAV.")
    note: Optional[str] = Field(None, description="Optional description/label.")
    language: Optional[str] = Field(None, description="Primary language of reference (optional).")


@app.post("/speakers/register", summary="Register (enroll) a speaker from URL or uploaded WAV")
@app.post("/speakers/register", summary="Register (enroll) a speaker via WAV upload only")
def register_speaker(
    wav_file: UploadFile = File(..., description="Reference WAV (mono, 3–10s)"),
    speaker_id: Optional[str] = Form(None, description="Desired ID; autogenerated if omitted"),
    note: Optional[str] = Form(None, description="Optional description/label"),
    language: Optional[str] = Form(None, description="Primary language of reference (optional)"),
):
    """
    Accepts only multipart/form-data with a WAV file.
    No URL-based registration is supported anymore.
    """
    # read file
    content = wav_file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty uploaded file.")

    # validate minimal header (optional: keep lightweight)
    # You can add stricter checks with soundfile if you want to assert WAV format.
    # import soundfile as sf
    # try:
    #     import io
    #     _data, _sr = sf.read(io.BytesIO(content), dtype="float32")
    # except Exception:
    #     raise HTTPException(status_code=400, detail="Invalid/unsupported audio file. Please upload a WAV.")

    # build speaker_id
    base = wav_file.filename or "uploaded"
    sid = (speaker_id or f"spk_{_hash_text(base)}").strip()

    # persist WAV + metadata
    meta = {
        "source": "upload",
        "original_filename": wav_file.filename,
    }
    if note:
        meta["note"] = note
    if language:
        meta["language"] = language

    info = _register_wav(speaker_id=sid, wav_bytes=content, meta=meta)
    return {"ok": True, "speaker_id": sid, "meta": info}


@app.get("/speakers", summary="List built-in (model) speakers and registered speakers")
def list_speakers_combined(model_name: Optional[str] = Query(None, description="Optional model to inspect")):
    # built-in speakers from the (optionally chosen) model
    builtin: List[str] = []
    try:
        tts = get_tts(model_name=model_name)
        spk = getattr(tts, "speakers", None)
        if isinstance(spk, (list, tuple)):
            builtin = list(spk)
    except Exception:
        builtin = []
    # registered speakers (our enrolled ones)
    registered = _load_registered_speakers()
    return {"model": (model_name or MODEL_NAME), "built_in": builtin, "registered": registered}


@app.get("/speakers/builtin", summary="List built-in (model) speakers only")
def list_speakers_builtin(model_name: Optional[str] = Query(None, description="Optional model to inspect")):
    try:
        tts = get_tts(model_name=model_name)
        spk = getattr(tts, "speakers", None)
        if isinstance(spk, (list, tuple)):
            return {"model": (model_name or MODEL_NAME), "speakers": list(spk)}
    except Exception:
        pass
    return {"model": (model_name or MODEL_NAME), "speakers": []}


@app.get("/speakers/registered", summary="List registered speakers only")
def list_speakers_registered():
    return {"speakers": _load_registered_speakers()}


@app.delete("/speakers/{speaker_id}", summary="Delete a registered speaker")
def delete_speaker(speaker_id: str = FPath(..., description="Speaker ID to delete")):
    wav = _speaker_path(speaker_id)
    meta = _speaker_meta_path(speaker_id)
    deleted = False
    if wav.exists():
        wav.unlink()
        deleted = True
    if meta.exists():
        meta.unlink()
        deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found.")
    return {"ok": True, "deleted": speaker_id}


# =========================
# Synthesis API
# =========================

class SynthesizeJSON(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field(None, description="Language code, e.g. 'de', 'en', 'fr'")
    speaker: Optional[str] = Field(None, description="Registered speaker_id or built-in speaker name (e.g., 'speaker_0')")
    speaker_wav_url: Optional[str] = Field(None, description="URL to reference WAV (used if no registered speaker)")
    sample_rate: Optional[int] = Field(None, description="WAV sample rate (default 24000)")
    model_name: Optional[str] = Field(None, description="Optional TTS model name to use instead of default")


def _download_to_tmp(url: str) -> str:
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


@app.post("/synthesize", response_class=StreamingResponse, summary="Synthesize via JSON")
def synthesize_json(payload: SynthesizeJSON):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    lang = (payload.language or DEFAULT_LANGUAGE).strip()
    sr = payload.sample_rate or DEFAULT_SAMPLE_RATE

    # Resolve speaker_wav precedence:
    speaker_wav_path = None
    speaker = (payload.speaker or "").strip()

    # If speaker refers to a registered speaker, map to stored WAV
    if speaker and speaker.startswith("spk_"):
        wav = _ensure_registered(speaker)
        speaker_wav_path = str(wav)

    # Else allow built-in speaker names like "speaker_0" (no wav file), or clone via URL
    if not speaker_wav_path and payload.speaker_wav_url:
        speaker_wav_path = _download_to_tmp(payload.speaker_wav_url)
        if not speaker:
            # auto-named temporary speaker (not persisted)
            speaker = "clone_" + _hash_text(payload.speaker_wav_url)

    # If still nothing provided, try to fall back to a built-in default speaker
    if not speaker and not speaker_wav_path:
        # Use built-in first available, typically "speaker_0"
        try:
            tts_probe = get_tts(model_name=payload.model_name)
            speakers = getattr(tts_probe, "speakers", None)
            if isinstance(speakers, (list, tuple)) and len(speakers) > 0:
                speaker = str(speakers[0])
            else:
                raise HTTPException(status_code=400, detail="Provide 'speaker' (e.g., 'speaker_0' or a registered 'spk_*') or 'speaker_wav_url'.")
        except Exception:
            raise HTTPException(status_code=400, detail="Provide 'speaker' (e.g., 'speaker_0' or a registered 'spk_*') or 'speaker_wav_url'.")

    try:
        with _infer_lock:
            tts = get_tts(model_name=payload.model_name)
            wav = tts.tts(
                text=text,
                speaker=speaker or None,
                speaker_wav=speaker_wav_path,
                language=lang,
            )

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        # cleanup temp files
        if speaker_wav_path and speaker_wav_path.startswith("/tmp") and os.path.exists(speaker_wav_path):
            try:
                os.remove(speaker_wav_path)
            except Exception:
                pass


@app.post("/synthesize-multipart", response_class=StreamingResponse, summary="Synthesize with multipart/form-data")
def synthesize_multipart(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(None),
    speaker: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None, description="WAV for cloning (if no registered speaker)"),
):
    # Prepare payload-like fields
    lang = (language or DEFAULT_LANGUAGE).strip()
    sr = sample_rate or DEFAULT_SAMPLE_RATE
    spk = (speaker or "").strip()
    model_override = (model_name or None)

    # write uploads to tmp
    tmp_speaker = None
    if speaker_wav is not None:
        fd, tmp_speaker = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        with open(tmp_speaker, "wb") as f:
            f.write(speaker_wav.file.read())
        if not spk:
            spk = "clone_" + _hash_text(getattr(speaker_wav, "filename", "upload"))

    # Resolve precedence (registered first, then uploaded, then fallback built-in)
    speaker_wav_path = None
    if spk and spk.startswith("spk_"):
        speaker_wav_path = str(_ensure_registered(spk))
    elif tmp_speaker:
        speaker_wav_path = tmp_speaker

    if not spk and not speaker_wav_path:
        # fallback to built-in
        tts_probe = get_tts(model_name=model_override)
        speakers = getattr(tts_probe, "speakers", None)
        if isinstance(speakers, (list, tuple)) and len(speakers) > 0:
            spk = str(speakers[0])
        else:
            raise HTTPException(status_code=400, detail="Provide 'speaker' or 'speaker_wav'.")

    try:
        with _infer_lock:
            tts = get_tts(model_name=model_override)
            wav = tts.tts(
                text=text.strip(),
                speaker=spk or None,
                speaker_wav=speaker_wav_path,
                language=lang,
            )

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        if tmp_speaker and os.path.exists(tmp_speaker):
            try:
                os.remove(tmp_speaker)
            except Exception:
                pass


@app.get("/")
def root():
    return JSONResponse({
        "name": "XTTS TTS Server",
        "defaults": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "language": DEFAULT_LANGUAGE,
            "sample_rate": DEFAULT_SAMPLE_RATE
        },
        "health": "/health",
        "openapi": "/openapi.json",
        "docs": "/docs",
        "endpoints": {
            "models": {
                "GET /models": "List loaded models",
                "POST /models/pull": {"model_name": "tts_models/.../xtts_v2", "device?": "cuda|cpu"},
                "DELETE /models/evict": {"model_name": "tts_models/.../xtts_v2", "device?": "cuda|cpu"}
            },
            "register_speaker": {
                "POST /speakers/register": {
                    "json": {"speaker_id?": "spk_myvoice", "wav_url?": "https://.../ref.wav", "note?": "...", "language?": "de"},
                    "multipart": ["wav_file", "speaker_id?"]
                }
            },
            "list_speakers": "GET /speakers?model_name=tts_models/... (optional)",
            "delete_speaker": "DELETE /speakers/{speaker_id}",
            "synthesize_json": {
                "POST /synthesize": {
                    "text": "Hallo Welt",
                    "language": "de",
                    "speaker?": "speaker_0 or spk_...",
                    "speaker_wav_url?": "https://.../ref.wav",
                    "sample_rate?": 24000,
                    "model_name?": "tts_models/multilingual/multi-dataset/xtts_v2"
                }
            },
            "synthesize_multipart": {
                "POST /synthesize-multipart": ["text", "language?", "sample_rate?", "speaker?", "model_name?", "speaker_wav? (file)"]
            }
        }
    })
