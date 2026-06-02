FROM python:3.11-slim

WORKDIR /app

# Install dependencies from backend/requirements.txt
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Railway sets $PORT automatically; default to 8000 for local
ENV PORT=8000

EXPOSE $PORT

# MUST use shell form (not exec form) so $PORT gets expanded
CMD uvicorn backend.api:app --host 0.0.0.0 --port $PORT
