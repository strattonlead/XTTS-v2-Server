import io
import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path as FPath, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf

# prevent CPML prompt in non-interactive envs
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from TTS.api import TTS

app = FastAPI(
    title="XTTS-v2 TTS Server",
    version="2.0.0",
    description="XTTS-v2 TTS with speaker enrollment (persistent), CUDA-ready."
)

# --- Configuration
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "de")
DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "24000"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()  # persistent volume
SPEAKERS_DIR = DATA_DIR / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

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
        "model": MODEL_NAME,
        "data_dir": str(DATA_DIR),
        "registered_speakers": _load_registered_speakers(),
    }


# =========================
# Speaker Enrollment API
# =========================

class RegisterSpeakerJSON(BaseModel):
    speaker_id: Optional[str] = Field(None, description="Desired ID; if omitted, generated from source.")
    wav_url: Optional[str] = Field(None, description="HTTP(s) URL to reference WAV.")
    note: Optional[str] = Field(None, description="Optional description/label.")
    language: Optional[str] = Field(None, description="Primary language of reference (optional).")


@app.post("/speakers/register", summary="Register (enroll) a speaker from URL or uploaded WAV")
def register_speaker(
    payload: Optional[RegisterSpeakerJSON] = None,
    wav_file: Optional[UploadFile] = File(None, description="Upload reference WAV instead of URL"),
):
    """
    Provide either JSON with `wav_url` or a multipart upload `wav_file`.
    Returns metadata with the final speaker_id.
    """
    if payload is None and wav_file is None:
        raise HTTPException(status_code=400, detail="Provide JSON (with wav_url) or a wav_file upload.")

    # Determine source & speaker_id
    meta: Dict = {}
    wav_bytes: Optional[bytes] = None

    if wav_file is not None:
        content = wav_file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty uploaded file.")
        wav_bytes = content
        base = wav_file.filename or "uploaded"
        speaker_id = (payload.speaker_id if payload else None) or f"spk_{_hash_text(base)}"
        meta.update({
            "source": "upload",
            "original_filename": wav_file.filename,
        })
        if payload:
            if payload.note: meta["note"] = payload.note
            if payload.language: meta["language"] = payload.language

    else:
        if not payload or not payload.wav_url:
            raise HTTPException(status_code=400, detail="Missing wav_url in JSON or wav_file in multipart.")
        import requests
        try:
            r = requests.get(payload.wav_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download wav_url: {e}")
        wav_bytes = r.content
        speaker_id = payload.speaker_id or f"spk_{_hash_text(payload.wav_url)}"
        meta.update({
            "source": "url",
            "wav_url": payload.wav_url,
        })
        if payload.note: meta["note"] = payload.note
        if payload.language: meta["language"] = payload.language

    # Persist
    info = _register_wav(speaker_id=speaker_id, wav_bytes=wav_bytes, meta=meta)
    return {"ok": True, "speaker_id": speaker_id, "meta": info}


@app.get("/speakers", summary="List built-in (model) speakers and registered speakers")
def list_speakers_combined():
    # built-in speakers from the model (if provided by this TTS version)
    try:
        tts = get_tts()
        builtin = getattr(tts, "speakers", None)
        if isinstance(builtin, (list, tuple)):
            builtin = list(builtin)
        else:
            builtin = []  # some TTS versions don't expose this
    except Exception:
        builtin = []
    # registered speakers (our enrolled ones)
    registered = _load_registered_speakers()
    return {"built_in": builtin, "registered": registered}


@app.get("/speakers/builtin", summary="List built-in (model) speakers only")
def list_speakers_builtin():
    try:
        tts = get_tts()
        builtin = getattr(tts, "speakers", None)
        if isinstance(builtin, (list, tuple)):
            return {"speakers": list(builtin)}
    except Exception:
        pass
    return {"speakers": []}


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
    style_wav_url: Optional[str] = Field(None, description="Optional style reference WAV (prosody)")


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
            tts = get_tts()
            speakers = getattr(tts, "speakers", None)
            if isinstance(speakers, (list, tuple)) and len(speakers) > 0:
                speaker = str(speakers[0])
            else:
                # Last resort: require explicit input
                raise HTTPException(status_code=400, detail="Provide 'speaker' (e.g., 'speaker_0' or a registered 'spk_*') or 'speaker_wav_url'.")
        except Exception:
            raise HTTPException(status_code=400, detail="Provide 'speaker' (e.g., 'speaker_0' or a registered 'spk_*') or 'speaker_wav_url'.")

    # Optional style wav
    style_wav_path = None
    if payload.style_wav_url:
        style_wav_path = _download_to_tmp(payload.style_wav_url)

    try:
        with _infer_lock:
            tts = get_tts()
            wav = tts.tts(
                text=text,
                speaker=speaker or None,
                speaker_wav=speaker_wav_path,
                language=lang,
                #style_wav=style_wav_path,
            )

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        # cleanup temp files
        for p in [speaker_wav_path, style_wav_path]:
            if p and p.startswith("/tmp") and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.post("/synthesize-multipart", response_class=StreamingResponse, summary="Synthesize with multipart/form-data")
def synthesize_multipart(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(None),
    speaker: Optional[str] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None, description="WAV for cloning (if no registered speaker)"),
    #style_wav: Optional[UploadFile] = File(None, description="Optional style reference WAV"),
):
    payload = SynthesizeJSON(
        text=text,
        language=language,
        sample_rate=sample_rate,
        speaker=speaker,
        speaker_wav_url=None,
        #style_wav_url=None,
    )

    # write uploads to tmp and delegate to same logic
    tmp_speaker = None
    tmp_style = None
    if speaker_wav is not None:
        fd, tmp_speaker = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        with open(tmp_speaker, "wb") as f: f.write(speaker_wav.file.read())
        # Use a temporary speaker name if not provided
        if not payload.speaker:
            payload.speaker = "clone_" + _hash_text(getattr(speaker_wav, "filename", "upload"))
    if style_wav is not None:
        fd, tmp_style = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        with open(tmp_style, "wb") as f: f.write(style_wav.file.read())

    # Temporarily pass file paths using URLs fields (hacking through same function)
    try:
        # Call the same core logic but bypassing extra download
        lang = (payload.language or DEFAULT_LANGUAGE).strip()
        sr = payload.sample_rate or DEFAULT_SAMPLE_RATE

        # Resolve speaker wav precedence
        speaker_wav_path = None
        speaker = (payload.speaker or "").strip()

        if speaker and speaker.startswith("spk_"):
            wav_p = _ensure_registered(speaker)
            speaker_wav_path = str(wav_p)
        elif tmp_speaker:
            speaker_wav_path = tmp_speaker

        if not speaker and not speaker_wav_path:
            # fallback to built-in
            tts = get_tts()
            speakers = getattr(tts, "speakers", None)
            if isinstance(speakers, (list, tuple)) and len(speakers) > 0:
                speaker = str(speakers[0])
            else:
                raise HTTPException(status_code=400, detail="Provide 'speaker' or 'speaker_wav'.")

        with _infer_lock:
            tts = get_tts()
            wav = tts.tts(
                text=text.strip(),
                speaker=speaker or None,
                speaker_wav=speaker_wav_path,
                language=lang,
                #style_wav=tmp_style,
            )

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
                                 headers={"Content-Disposition": "inline; filename=tts.wav"})
    finally:
        for p in [tmp_speaker, tmp_style]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass


@app.get("/")
def root():
    return JSONResponse({
        "name": "XTTS-v2 TTS Server",
        "health": "/health",
        "openapi": "/openapi.json",
        "docs": "/docs",
        "endpoints": {
            "register_speaker": {
                "POST /speakers/register": {
                    "json": {"speaker_id?": "spk_myvoice", "wav_url?": "https://.../ref.wav", "note?": "...", "language?": "de"},
                    "multipart": ["wav_file", "speaker_id?"]
                }
            },
            "list_speakers": "GET /speakers",
            "delete_speaker": "DELETE /speakers/{speaker_id}",
            "synthesize_json": {
                "POST /synthesize": {
                    "text": "Hallo Welt",
                    "language": "de",
                    "speaker?": "speaker_0 or spk_...",
                    "speaker_wav_url?": "https://.../ref.wav",
                    "style_wav_url?": "https://.../style.wav",
                    "sample_rate?": 24000
                }
            },
            "synthesize_multipart": {
                "POST /synthesize-multipart": ["text", "language?", "sample_rate?", "speaker?", "speaker_wav? (file)", "style_wav? (file)"]
            }
        }
    })
