FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app.py utilities.py wsgi.py ./
COPY resnet-50-MRI.json ResUNet-MRI.json weights.hdf5 weights_seg.hdf5 ./
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "wsgi:app"]
