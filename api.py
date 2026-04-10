import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid

from infer import run_inference_api

# ── paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── app ──────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ──────────────────────────────────────────
def generate_job_id():
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

# ── routes ───────────────────────────────────────────

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    1. Save file
    2. Run inference
    3. Return job_id + output folder
    """

    # ── validate ─────────────────────────────────────
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
        return {"error": "Unsupported format"}

    # ── create job ───────────────────────────────────
    job_id = generate_job_id()

    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id

    job_upload_dir.mkdir(parents=True)
    job_output_dir.mkdir(parents=True)

    # ── save file ────────────────────────────────────
    file_path = job_upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ── run inference ────────────────────────────────
    run_inference_api(
        input_path=str(file_path),
        output_dir=str(job_output_dir)
    )

    # ── return metadata ──────────────────────────────
    return {
        "job_id": job_id,
        "input_file": str(file_path),
        "output_folder": str(job_output_dir),
        "files": os.listdir(job_output_dir)
    }


@app.get("/history")
def get_history():
    """
    List all jobs
    """
    jobs = []

    for job in sorted(os.listdir(OUTPUT_DIR)):
        job_path = OUTPUT_DIR / job
        jobs.append({
            "job_id": job,
            "files": os.listdir(job_path)
        })

    return {"jobs": jobs}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """
    Get specific job outputs
    """
    job_path = OUTPUT_DIR / job_id

    if not job_path.exists():
        return {"error": "Job not found"}

    return {
        "job_id": job_id,
        "files": os.listdir(job_path),
        "url_base": f"http://localhost:8001/outputs/{job_id}"
    }


# ── run ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)