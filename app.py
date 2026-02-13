import os
import math
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


# -------------------------------------------------------------------
# MODELOS PARA /music
# -------------------------------------------------------------------
class MusicReq(BaseModel):
    prompt: str
    duration: float = 4.0


class MusicRes(BaseModel):
    prompt: str
    duration: float
    audio_path: str
    sampling_rate: int


# -------------------------------------------------------------------
# MODELOS PARA /music-loop-from-wav
# -------------------------------------------------------------------
class MusicLoopReq(BaseModel):
    # Nombre del archivo base dentro de /outputs
    # Ej: "musicgen_20251124_173111.wav"
    filename: str

    # Duración objetivo (segundos). Por defecto ~10 minutos
    target_seconds: float = 600.0

    # Crossfade entre loops (ms)
    fade_ms: int = 400

    # Si quieres forzar nº de loops, pon un valor > 0.
    # Si es 0, se calcula automáticamente a partir de target_seconds.
    loops: int = 0


class MusicLoopRes(BaseModel):
    input_path: str
    output_path: str
    base_duration: float
    total_duration: float
    sampling_rate: int
    loops: int
    fade_ms: int


# -------------------------------------------------------------------
# HELPERS PARA AUDIO (loop desde WAV)
# -------------------------------------------------------------------
def load_audio(path: Path):
    """Lee un WAV y lo devuelve como float32 en [-1, 1]."""
    sr, data = wav.read(str(path))

    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return sr, data


def crossfade_loop(segment: np.ndarray, loops: int, fade_ms: int, sr: int) -> np.ndarray:
    """
    Repite 'segment' 'loops' veces aplicando crossfade entre ellas.
    Soporta mono (N,) y multicanal (N, C).
    """
    if loops < 1:
        raise ValueError("loops debe ser >= 1")

    mono = False
    if segment.ndim == 1:
        mono = True
        segment = segment[:, None]  # (N,1)

    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples < 0:
        fade_samples = 0

    out = segment.copy()

    for _ in range(loops - 1):
        s = segment

        if fade_samples > 0:
            f = min(fade_samples, out.shape[0], s.shape[0])
        else:
            f = 0

        if f > 0:
            fade_out = np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
            fade_in = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]

            out_tail = out[-f:] * fade_out
            s_head = s[:f] * fade_in

            mixed = out_tail + s_head

            out = np.vstack([out[:-f], mixed, s[f:]])
        else:
            out = np.vstack([out, s])

    if mono:
        out = out[:, 0]

    return out


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": "cuda:0" if device == 0 else "cpu",
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
            generate_kwargs={"max_new_tokens": tokens},
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
        sampling_rate=sr,
    )


@app.post("/music-loop-from-wav", response_model=MusicLoopRes)
def music_loop_from_wav(req: MusicLoopReq):
    """
    Crea un loop largo a partir de un WAV existente en /outputs.
    """
    if req.target_seconds <= 0:
        raise HTTPException(400, "target_seconds debe ser > 0")

    in_path = OUTPUT_DIR / req.filename
    if not in_path.exists():
        raise HTTPException(404, f"No existe el archivo base: {in_path}")

    try:
        sr, segment = load_audio(in_path)
    except Exception as e:
        raise HTTPException(500, f"Error leyendo WAV base: {str(e)}")

    base_duration = len(segment) / sr
    if base_duration <= 0:
        raise HTTPException(500, "La duración del archivo base es 0 segundos.")

    # Cálculo de loops
    if req.loops > 0:
        loops = req.loops
    else:
        loops = max(1, math.ceil(req.target_seconds / base_duration))

    try:
        out = crossfade_loop(segment, loops=loops, fade_ms=req.fade_ms, sr=sr)
    except Exception as e:
        raise HTTPException(500, f"Error generando loop: {str(e)}")

    total_duration = len(out) / sr

    # Normalizar y guardar
    peak = np.max(np.abs(out)) + 1e-9
    out = (out / peak).astype(np.float32)

    out_i16 = (out * 32767).clip(-32768, 32767).astype(np.int16)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"musicgen_loop_from_wav_{ts}.wav"

    try:
        wav.write(str(out_path), sr, out_i16)
    except Exception as e:
        raise HTTPException(500, f"Error guardando WAV de salida: {str(e)}")

    return MusicLoopRes(
        input_path=str(in_path),
        output_path=str(out_path),
        base_duration=base_duration,
        total_duration=total_duration,
        sampling_rate=sr,
        loops=loops,
        fade_ms=req.fade_ms,
    )
