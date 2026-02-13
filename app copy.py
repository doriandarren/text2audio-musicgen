import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import scipy.io.wavfile
from transformers import pipeline

app = FastAPI()

OUTPUT_DIR = Path("/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-medium")
print(f"🎵 Loading MusicGen model: {MODEL_ID}")

# GPU si está disponible, si no CPU
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
    duration: float = 4.0  # segundos aproximados

class MusicRes(BaseModel):
    prompt: str
    duration: float
    audio_path: str
    sampling_rate: int

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": "cuda:0" if device == 0 else "cpu"}

@app.post("/music", response_model=MusicRes)
def music(req: MusicReq):
    if req.duration <= 0:
        raise HTTPException(400, "duration debe ser > 0")

    # Duración aproximada por tokens (ajusta factor si quieres)
    tokens = max(32, int(req.duration * 25))

    try:
        out = synth(
            req.prompt,
            generate_kwargs={"max_new_tokens": tokens}  # ✅ forma correcta
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory. Baja duration o usa musicgen-small."
            )
        raise HTTPException(500, f"Error en el modelo: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Error en el modelo: {str(e)}")

    audio = out["audio"]
    sr = out["sampling_rate"]

    # A veces viene como shape (1, N). Nos quedamos con el primer canal.
    if hasattr(audio, "ndim") and audio.ndim == 2:
        audio = audio[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"musicgen_{ts}.wav"
    scipy.io.wavfile.write(str(path), sr, audio)

    return MusicRes(
        prompt=req.prompt,
        duration=req.duration,
        audio_path=str(path),
        sampling_rate=sr
    )
