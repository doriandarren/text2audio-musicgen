## pip install pydub
from pathlib import Path
from pydub import AudioSegment

OUT_DIR = Path("outputs")  # tu carpeta host montada a /outputs
CROSSFADE_MS = 800         # 0.8s suele quedar fino

# busca todos los chunks
files = sorted(
    list(OUT_DIR.glob("musicgen_*.wav")) +
    list(OUT_DIR.glob("musicgen_chunk_*.wav"))
)

if not files:
    raise SystemExit("No encuentro chunks en outputs/")

print("Chunks encontrados:")
for f in files:
    print(" -", f.name)

final = AudioSegment.from_wav(files[0])
for f in files[1:]:
    seg = AudioSegment.from_wav(f)
    final = final.append(seg, crossfade=CROSSFADE_MS)

out_path = OUT_DIR / "final_long.wav"
final.export(out_path, format="wav")
print("✅ Generado:", out_path)
print("Duración total (s):", len(final)/1000)
