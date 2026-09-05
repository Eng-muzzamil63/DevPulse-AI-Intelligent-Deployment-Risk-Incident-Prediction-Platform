FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app/backend
COPY ml /app/ml
CMD ["sh","-c","python /app/ml/train.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
