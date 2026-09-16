# linux/amd64 pinned: paddlepaddle publishes no arm64 CPU wheels, so on Apple Silicon
# this must build under emulation rather than fail at pip install time.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Skips a network round-trip to the model hosts on every process start.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_MODEL_SOURCE=huggingface

WORKDIR /app

# libgl1 + libglib2.0-0 are OpenCV runtime deps; libgomp1 is Paddle's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the OCR weights into the image. Without this every fresh container re-downloads
# ~100MB from the model host on first request, which also makes eval runs non-reproducible
# and fails outright on an offline machine.
COPY scripts/warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
