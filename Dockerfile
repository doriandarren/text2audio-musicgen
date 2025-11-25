FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ✅ Torch >= 2.6 (pip cogerá la wheel Linux cu124)
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
RUN mkdir -p /outputs /app/hf_cache

EXPOSE 7880
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7880"]
