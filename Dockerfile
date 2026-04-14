# ── Stage 1: build dependencies ──────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies ke folder terpisah agar layer cache efisien
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────
FROM python:3.10-slim

WORKDIR /app

# Salin installed packages dari builder
COPY --from=builder /install /usr/local

# Salin source code & data
COPY app.py .
COPY data/ ./data/

EXPOSE 8050

# Gunicorn: 2 worker, timeout 120s (data loading bisa lambat saat cold start)
CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
