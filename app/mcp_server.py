# mcp_server.py
import os, json, hashlib, tempfile
from pathlib import Path
from typing import Optional
from mcp.server import Server
from mcp.types import Tool, TextContent

DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
SPEAKERS_DIR = DATA_DIR / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

def _speaker_path(s): return SPEAKERS_DIR / f"{s}.wav"
def _hash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

srv = Server("xtts-mcp")

@srv.tool("register_speaker", "Register a speaker from WAV bytes (base64) or URL")
def register_speaker(args: dict):
    speaker_id = args.get("speaker_id") or f"spk_{_hash(args.get('hint') or 'x')}"
    if "wav_b64" in args:
        import base64
        content = base64.b64decode(args["wav_b64"])
        _speaker_path(speaker_id).write_bytes(content)
    elif "wav_url" in args:
        import requests
        r = requests.get(args["wav_url"], timeout=30); r.raise_for_status()
        _speaker_path(speaker_id).write_bytes(r.content)
    else:
        raise ValueError("Provide wav_b64 or wav_url")
    return [TextContent(text=json.dumps({"ok": True, "speaker_id": speaker_id}))]

@srv.tool("list_speakers", "List registered speakers")
def list_speakers(args: dict):
    lst = sorted([p.stem for p in SPEAKERS_DIR.glob("*.wav")])
    return [TextContent(text=json.dumps({"speakers": lst}))]

@srv.tool("synthesize", "Synthesize text using an enrolled or built-in speaker")
def synthesize(args: dict):
    """
    args: {"text": str, "language": "de", "speaker": "spk_..." or "speaker_0", "sample_rate": 24000}
    returns: base64 wav
    """
    import soundfile as sf, io, base64
    from TTS.api import TTS
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    model = args.get("model_name") or "tts_models/multilingual/multi-dataset/xtts_v2"
    device = args.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    tts = TTS(model).to(device)
    speaker = args.get("speaker")
    speaker_wav = None
    if speaker and speaker.startswith("spk_"):
        p = _speaker_path(speaker)
        if not p.exists(): raise ValueError("Unknown speaker_id")
        speaker_wav = str(p)

    wav = tts.tts(text=args["text"], language=args.get("language"), speaker=speaker, speaker_wav=speaker_wav)
    buf = io.BytesIO()
    sr = int(args.get("sample_rate") or 24000)
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16"); buf.seek(0)
    return [TextContent(text=json.dumps({"wav_b64": base64.b64encode(buf.read()).decode()}))]

if __name__ == "__main__":
    srv.run_stdio()  # MCP over stdio
