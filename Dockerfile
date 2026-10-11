FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
        python -m pip install \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements.txt

COPY backend ./backend
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host \"$BACKEND_HOST\" --port \"$BACKEND_PORT\""]
