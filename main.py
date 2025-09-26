from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from TTS.api import TTS
import torch, os, io, glob, shutil, tempfile
import soundfile as sf
import numpy as np

app = FastAPI(title="XTTS Server (coqui-tts)", version="1.0.0")

# ---- Config ----
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
VOICES_DIR = os.getenv("VOICES_DIR", "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

device_env = os.getenv("DEVICE")
device = device_env if device_env in {"cpu", "cuda"} else ("cuda" if torch.cuda.is_available() else "cpu")

ISO3_TO_ISO2 = {"deu": "de", "eng": "en"}
def _lang_norm(l: str) -> str: return ISO3_TO_ISO2.get(l.lower(), l.lower())

def _collect_voice_samples(name: Optional[str]) -> Optional[List[str]]:
    if not name: return None
    d = os.path.join(VOICES_DIR, name)
    if not os.path.isdir(d):
        raise HTTPException(404, f"Voice '{name}' not found.")
    files = sorted(glob.glob(os.path.join(d, "*.wav")))
    if not files:
        raise HTTPException(400, f"No .wav files in voice '{name}'.")
    return files

# ---- Robust loader (mit Fallback für Torch>=2.6 pickle-safe) ----
def load_xtts():
    try:
        return TTS(MODEL_NAME).to(device)
    except Exception as e:
        # Fallback: Allowlist relevanter Klassen und nochmal versuchen
        import importlib, inspect
        from torch.serialization import add_safe_globals, safe_globals
        modules = [
            "TTS.tts.models.xtts",
            "TTS.tts.configs.xtts_config",
            "TTS.config.shared_configs",
            "TTS.tts.configs.shared_configs",
        ]
        allowed = []
        for name in modules:
            try:
                m = importlib.import_module(name)
                allowed += [obj for _, obj in inspect.getmembers(m, inspect.isclass)]
            except Exception:
                pass
        add_safe_globals(allowed)
        with safe_globals(allowed):
            return TTS(MODEL_NAME).to(device)

tts = load_xtts()

class TTSRequest(BaseModel):
    text: str
    lang: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    seed: Optional[int] = None
    split_sentences: Optional[bool] = True

@app.get("/health")
def health():
    cap = torch.cuda.get_device_capability(0) if (device=="cuda" and torch.cuda.is_available()) else None
    return {"status":"ok","device":device,"torch":torch.__version__,
            "cuda":getattr(torch.version,"cuda",None),"gpu_capability":cap,"model":MODEL_NAME}

@app.get("/voices")
def list_voices():
    out=[]
    for d in sorted(os.listdir(VOICES_DIR)):
        p=os.path.join(VOICES_DIR,d)
        if os.path.isdir(p):
            out.append({"name":d,"num_samples":len(glob.glob(os.path.join(p,"*.wav")))})
    return {"voices":out}

@app.post("/voices/{name}")
async def add_voice(name: str, files: List[UploadFile] = File(...)):
    d=os.path.join(VOICES_DIR,name); os.makedirs(d, exist_ok=True)
    n=0
    for i,f in enumerate(files,1):
        if not f.filename.lower().endswith(".wav"):
            raise HTTPException(400, f"Only .wav supported (got {f.filename}).")
        data=await f.read()
        try:
            a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as ex:
            raise HTTPException(400, f"Invalid WAV ({f.filename}): {ex}")
        sf.write(os.path.join(d,f"{i:02d}.wav"), a, sr, format="WAV", subtype="PCM_16")
        n+=1
    return {"ok":True,"voice":name,"saved":n}

@app.delete("/voices/{name}")
def delete_voice(name:str):
    d=os.path.join(VOICES_DIR,name)
    if not os.path.isdir(d): raise HTTPException(404, f"Voice '{name}' not found.")
    shutil.rmtree(d); return {"ok":True,"deleted":name}

@app.post("/tts")
def synth(req:TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Field 'text' must not be empty.")
    lang=_lang_norm(req.lang)
    if req.seed is not None:
        torch.manual_seed(int(req.seed)); np.random.seed(int(req.seed))

    refs = _collect_voice_samples(req.voice)
    speaker_arg = refs if (refs and len(refs)>1) else (refs[0] if refs else None)

    # XTTS: waveform zurückgeben
    wav = tts.tts(
        text=req.text.strip(),
        language=lang,
        speaker_wav=speaker_arg,
        speed=req.speed if req.speed else 1.0,
        split_sentences=bool(req.split_sentences),
    )
    sr = getattr(tts.synthesizer, "output_sample_rate", 24000)
    audio = np.asarray(wav, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    # To file (RAM-schonend) und als Download zurückgeben
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, audio, sr, format="WAV", subtype="PCM_16")
    filename = f"xtts-{lang}{('-'+req.voice) if req.voice else ''}.wav"
    return FileResponse(tmp.name, media_type="audio/wav", filename=filename)
