import io


# Write to WAV buffer at desired sample-rate
# (Coqui returns at the model's native rate; soundfile will resave; for resampling you could add librosa if desired)
buf = io.BytesIO()
sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
buf.seek(0)


headers = {
"Content-Disposition": "inline; filename=tts.wav"
}
return StreamingResponse(content=buf, media_type="audio/wav", headers=headers)
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
# Save upload to temp file path for the API
fd, tmp_path = tempfile.mkstemp(suffix=".wav")
os.close(fd)
with open(tmp_path, "wb") as f:
f.write(speaker_wav.file.read())


try:
with _infer_lock:
tts = get_tts()
wav = tts.tts(
text=text,
speaker_wav=tmp_path,
language=lang,
)


buf = io.BytesIO()
sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
buf.seek(0)


headers = {"Content-Disposition": "inline; filename=tts.wav"}
return StreamingResponse(content=buf, media_type="audio/wav", headers=headers)
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
"body": {
"text": "Hallo Welt",
"language": "de",
"speaker_wav_url": "https://example.com/ref.wav (optional)",
"sample_rate": 24000
}
},
"synthesize_multipart": {
"method": "POST",
"path": "/synthesize-multipart",
"form": ["text", "language?", "sample_rate?", "speaker_wav? (file)"]
}
})