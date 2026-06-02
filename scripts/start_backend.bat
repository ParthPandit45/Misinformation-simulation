@echo off
REM Start FastAPI backend on port 8000
cd backend
call uvicorn api:app --host 0.0.0.0 --port 8000
