import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline

app = FastAPI()

OUTPUT_DIR = Path("/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-medium")
print(f"🎵 Loading MusicGen model: {MODEL_ID}")

# GPU si existe
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if device == 0 else torch.float32

synth = pipeline(
    "text-to-audio",
    model=MODEL_ID,
    device=device,
    torch_dtype=dtype,
)

class MusicReq(BaseModel):
    prompt: str
    duration: float = 4.0

class MusicRes(BaseModel):
    prompt: str
    duration: float
    audio_path: str
    sampling_rate: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": "cuda:0" if device == 0 else "cpu"
    }

@app.post("/music", response_model=MusicRes)
def music(req: MusicReq):

    if req.duration <= 0:
        raise HTTPException(400, "duration debe ser > 0")

    # Duración aproximada: tokens = segundos * factor
    tokens = max(32, int(req.duration * 25))

    try:
        out = synth(
            req.prompt,
            generate_kwargs={"max_new_tokens": tokens}
        )
    except Exception as e:
        raise HTTPException(500, f"Error en el modelo: {str(e)}")

    audio = out["audio"]
    sr = out["sampling_rate"]

    # Si viene (1, N), tomar canal 0
    if hasattr(audio, "ndim") and audio.ndim == 2:
        audio = audio[0]

    # Normalizar para volumen correcto
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak

    # Convertir a int16 para guardar WAV
    audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"musicgen_{ts}.wav"

    wav.write(str(path), sr, audio_i16)

    return MusicRes(
        prompt=req.prompt,
        duration=req.duration,
        audio_path=str(path),
        sampling_rate=sr
    )
