FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# System deps:
# - tesseract-ocr: binary for pytesseract
# - libfreetype6: FreeType runtime
# - fontconfig + fonts: system fonts (diacritics etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libfreetype6 \
      fontconfig \
      fonts-noto \
      fonts-dejavu-core \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PaddlePaddle wheels are hosted on Paddle index; install it before paddleocr if needed.
# If you don't need paddleocr, you can remove the next RUN line.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "evaluate.py", "--help"]
