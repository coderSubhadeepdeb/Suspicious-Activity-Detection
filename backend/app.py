import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes import router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Automatically create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

import mimetypes
mimetypes.init()
mimetypes.add_type("video/mp4", ".mp4")

app = FastAPI(
    title="Video Anomaly Detection API",
    description="Upload surveillance video and get anomaly heatmap overlay",
    version="1.0.0"
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed output videos
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Serve frontend static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Video Anomaly Detection API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}