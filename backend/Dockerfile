FROM python:3.12-slim

WORKDIR /app

# Set default port for Railway (can be overridden by $PORT)
ENV PORT=8000

# Install build dependencies (if any) and runtime dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

EXPOSE $PORT

# Start FastAPI with uvicorn on the dynamic port
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]

# Start FastAPI with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
