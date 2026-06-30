import os
import sys
import uuid
import json
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from backend.utils import validate_video, generate_anomaly_report
from inference.detect import detect_anomalies
from inference.heatmap import generate_heatmap_video
from training.config import CHECKPOINT_DIR, OUTPUT_DIR

router = APIRouter()

# In-memory job store (use Redis in production)
jobs = {}

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_video_task(job_id: str, video_path: str):
    """Background task: run detection + generate heatmap video"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Running anomaly detection...'

        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = 'No trained model found. Train the model first.'
            return

        # Run detection
        results, threshold_low, threshold_high = detect_anomalies(video_path, checkpoint_path)

        jobs[job_id]['message'] = 'Generating heatmap video...'

        # Generate heatmap video
        output_filename = f"{job_id}_heatmap.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        generate_heatmap_video(results, threshold_low, threshold_high, output_path)

        # Generate anomaly report
        report = generate_anomaly_report(results, threshold_low, threshold_high)

        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['message'] = 'Processing complete'
        jobs[job_id]['result'] = {
            'heatmap_video_url': f"/outputs/{output_filename}",
            'report': report
        }

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
    finally:
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)


@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate file
    error = validate_video(file)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Save uploaded file
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create job entry
    jobs[job_id] = {
        'job_id': job_id,
        'filename': file.filename,
        'status': 'queued',
        'message': 'Job queued for processing',
        'created_at': datetime.now().isoformat(),
        'result': None
    }

    # Start background processing
    background_tasks.add_task(process_video_task, job_id, video_path)

    return JSONResponse(content={'job_id': job_id, 'status': 'queued'})


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JSONResponse(content={
        'job_id': job_id,
        'status': job['status'],
        'message': job['message']
    })


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")
    return JSONResponse(content={
        'job_id': job_id,
        'filename': job['filename'],
        'result': job['result']
    })


@router.get("/jobs")
async def list_jobs():
    return JSONResponse(content=list(jobs.values()))