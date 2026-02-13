# Notes

Huggingface module: facebook/musicgen-medium

## Instalación

```sh
Crear carpeta en la raíz: "outputs"

```

## Comandos utilizados

```sh

// Comando
docker compose down
docker compose up -d --build


// OR
docker compose down
docker compose build --no-cache
docker compose up -d+


docker logs musicgen-text2audio --tail=50



docker exec -it musicgen-medium-api bash




curl -X POST http://192.168.1.103:7880/music-continue \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"lo-fi hip hop beat with soft piano and vinyl crackle\",\"prev_audio_path\":\"/outputs/XXX.wav\",\"duration\":6}"




curl -X POST http://192.168.1.103:7880/music-continue \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"lo-fi hip hop beat with soft piano and vinyl crackle\",
    \"prev_audio_path\": \"/outputs/musicgen_20251121_153728.wav\",
    \"duration\": 6
  }"




//Crea archivo:


  python3 loop_from_wav.py \
  --input /outputs/musicgen_20251123_195102.wav \
  --target-seconds 600 \
  --fade-ms 400


```
