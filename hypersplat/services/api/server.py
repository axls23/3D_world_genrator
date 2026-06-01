"""
3DGS Demo Server with Level 2 ACE-Zero Pipeline & Job Queue

Features:
- Video upload and processing
- Job Queue and State Machine (queued, running, processing, completed, failed, cancelled)
- WebSocket-based real-time updates & log streaming
- Checkpoint management and viewing
- Interactive 3D viewer integration

API Endpoints:
  POST   /api/upload        - Upload video file
  POST   /api/jobs/submit   - Submit training job to queue
  GET    /api/jobs/queue    - List all jobs in queue
  GET    /api/jobs/{id}     - Get specific job details
  GET    /api/jobs/{id}/logs- Get paginated logs for a job
  DELETE /api/jobs/{id}     - Cancel/delete job
  POST   /api/train         - Start training pipeline (legacy alias for submit)
  POST   /api/stop          - Stop training (legacy alias for cancel active)
  GET    /api/status        - Get training status (legacy alias for active job status)
  GET    /api/checkpoints   - List available checkpoints
  GET    /api/ply           - List PLY files
  POST   /api/view          - Start 3D viewer
  WS     /api/ws            - Real-time updates (WebSocket)
"""

import os
import shutil
import subprocess
import asyncio
import json
import uvicorn
import datetime
import time
import glob
import collections
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Optional, Dict, Set
from pydantic import BaseModel

# Suppress noisy /api/status polling logs
import logging

class StatusEndpointFilter(logging.Filter):
    """Filter out repetitive /api/status log entries."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(StatusEndpointFilter())

app = FastAPI(
    title="3DGS Demo Server",
    description="Video to 3D Gaussian Splatting with ACE-Zero Pipeline and Job Queue",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories - All outputs consolidated in demo/output/
CURRENT_DIR = Path(__file__).parent.resolve()
STATIC_DIR = CURRENT_DIR / "static"
GSPLAT_ROOT = CURRENT_DIR.parent.parent.parent
SCRIPTS_DIR = GSPLAT_ROOT / "scripts" / "pipeline"

# Single unified output directory
OUTPUT_DIR = GSPLAT_ROOT / "demo" / "output"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = OUTPUT_DIR / "results"
ACEZERO_DIR = OUTPUT_DIR / "acezero"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ACEZERO_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def get_python_executable() -> str:
    # Prefer the 3dgrut environment python if available on Windows
    conda_env_python = "C:\\Users\\sxhil_25660\\anaconda3\\envs\\3dgrut\\python.exe"
    if os.path.exists(conda_env_python):
        return conda_env_python
    return sys.executable or "python"


class AppState:
    """Application state management with asynchronous Job Queue and WebSocket Pub-Sub"""
    def __init__(self):
        self.jobs: Dict[str, dict] = {}  # job_id -> job dict
        self.job_ids_queue: List[str] = []  # List of pending/queued job IDs
        self.active_job_id: Optional[str] = None
        
        self.viewer_process = None
        self.viewer_url = None
        self.process_type = None  # "viewer" or None
        self.process = None  # Active viewer process
        
        # WebSocket Pub-Sub Registry
        self.ws_subscribers: Dict[str, Set[WebSocket]] = collections.defaultdict(set)  # job_id -> WebSockets
        self.global_ws_subscribers: Set[WebSocket] = set()  # WebSockets connected to root
        
        # Async worker loop task
        self.worker_task: Optional[asyncio.Task] = None
        
        self.current_video_path = None
        
    def enqueue_job(self, video_path: str, config: dict) -> str:
        import uuid
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued",
            "video_path": video_path,
            "filename": Path(video_path).name,
            "config": config,
            "progress": 0,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "completed_at": None,
            "logs": ["Job enqueued. Waiting for worker process..."],
            "error": None,
            "process": None,
            "training_viewer_url": None
        }
        self.jobs[job_id] = job
        self.job_ids_queue.append(job_id)
        
        # Launch worker loop if not already running
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.worker_loop())
            
        return job_id
        
    async def worker_loop(self):
        print("[Worker] Daemon loop started.")
        while self.job_ids_queue:
            job_id = self.job_ids_queue.pop(0)
            job = self.jobs.get(job_id)
            if not job or job["status"] == "cancelled":
                continue
                
            self.active_job_id = job_id
            try:
                await self.run_job(job)
            except Exception as e:
                import traceback
                print(f"[Worker] Error running job {job_id}: {e}")
                traceback.print_exc()
            finally:
                self.active_job_id = None
        print("[Worker] Daemon loop idle.")
        
    async def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
            
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if job_id in self.job_ids_queue:
                self.job_ids_queue.remove(job_id)
            job["logs"].append("[System] Job cancelled in queue.")
            await self.broadcast_status(job)
            await self.broadcast_log(job, "[System] Job cancelled in queue.")
            return True
            
        if job["status"] in ["running", "processing"]:
            job["status"] = "cancelled"
            job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            proc = job.get("process")
            if proc:
                try:
                    proc.terminate()
                except:
                    pass
            job["logs"].append("[System] Active job cancelled by user request.")
            await self.broadcast_status(job)
            await self.broadcast_log(job, "[System] Active job cancelled by user request.")
            return True
            
        return False
        
    async def run_job(self, job):
        job["status"] = "running"
        job["started_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await self.broadcast_status(job)
        await self.broadcast_queue_update()
        
        # Build training command
        video_path = job["video_path"]
        config = job["config"]
        
        pipeline_script = SCRIPTS_DIR / "automated_intelligent_pipeline.py"
        
        cmd = [
            get_python_executable(), "-u", str(pipeline_script),
            video_path,
            "--output_dir", str(OUTPUT_DIR),
            "--fps", str(config.get("fps", 2.0)),
            "--max_steps", str(config.get("max_steps", 7000)),
            "--data_factor", str(config.get("data_factor", 2)),
        ]
        
        if config.get("with_ut"):
            cmd.append("--with_ut")
        if config.get("with_eval3d"):
            cmd.append("--with_eval3d")
            
        if config.get("use_zero123") or config.get("genvs"):
            cmd.append("--genvs")
            num_views = config.get("genvs_views", 20)
            cmd.extend(["--genvs-views", str(num_views)])
            
        if not config.get("prune", True):
            cmd.append("--no-prune")
            
        if config.get("unified_stream") or config.get("streaming"):
            cmd.append("--streaming")
            
        quality_mode = config.get("quality_mode", "balanced")
        cmd.extend(["--quality-mode", quality_mode])
        
        min_conf = config.get("min_registration_confidence", 1000)
        cmd.extend(["--min-registration-confidence", str(min_conf)])
        
        depth_model = config.get("depth_model", "depth_anything")
        cmd.extend(["--depth-model", depth_model])

        if "sh_degree" in config:
            cmd.extend(["--sh_degree", str(config["sh_degree"])])
        if "means_lr" in config:
            cmd.extend(["--means_lr", str(config["means_lr"])])
        if "opacity_reg" in config:
            cmd.extend(["--opacity_reg", str(config["opacity_reg"])])
        if "scale_reg" in config:
            cmd.extend(["--scale_reg", str(config["scale_reg"])])
        if "ssim_lambda" in config:
            cmd.extend(["--ssim_lambda", str(config["ssim_lambda"])])
        if config.get("random_bkgd"):
            cmd.append("--random_bkgd")
        if config.get("pose_opt", True):
            cmd.append("--pose-opt")
        else:
            cmd.append("--no-pose-opt")
        if config.get("app_opt"):
            cmd.append("--app_opt")
            
        if config.get("early_stopping", True):
            cmd.append("--early-stopping")
            cmd.extend(["--early-stop-patience", str(config.get("early_stop_patience", 500))])
            cmd.extend(["--early-stop-min-delta", str(config.get("early_stop_min_delta", 0.001))])
            cmd.extend(["--early-stop-min-steps", str(config.get("early_stop_min_steps", 2000))])
        else:
            cmd.append("--no-early-stopping")

        # Skip ACE-Zero pose estimation if requested by the user, or if auto-detected and not explicitly overridden
        skip_ace_requested = config.get("skip_ace")
        expected_ace_output = OUTPUT_DIR / "acezero_output"
        sparse_dir = expected_ace_output / "sparse" / "0"
        has_existing_poses = (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists()
        
        if skip_ace_requested is True or (skip_ace_requested is None and has_existing_poses):
            cmd.append("--skip-ace")
            msg = "[System] Skipping Step 1 (Using existing camera poses from file)."
            job["logs"].append(msg)
            await self.broadcast_log(job, msg)
        else:
            msg = "[System] Executing Step 1 (ACE-Zero camera pose estimation)."
            job["logs"].append(msg)
            await self.broadcast_log(job, msg)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        msg = f"[System] Executing command: {' '.join(cmd)}"
        job["logs"].append(msg)
        await self.broadcast_log(job, msg)
        
        try:
            # Start process asynchronously
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(GSPLAT_ROOT),
                env=env
            )
            job["process"] = proc
            
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if decoded_line:
                    # Update progress and status
                    self.parse_progress(decoded_line, job)
                    
                    # Store and broadcast log
                    is_repetitive = ("it/s]" in decoded_line and "%" in decoded_line) or \
                                    ("Iteration" in decoded_line and "Loss" in decoded_line) or \
                                    ("Train Step" in decoded_line)
                                    
                    if is_repetitive and job["logs"] and self._is_repetitive_match(job["logs"][-1], decoded_line):
                        job["logs"][-1] = decoded_line
                    else:
                        job["logs"].append(decoded_line)
                        
                    await self.broadcast_log(job, decoded_line)
                    await self.broadcast_status(job)
            
            return_code = await proc.wait()
            
            if job["status"] == "cancelled":
                msg = "[System] Job was cancelled by user."
                job["logs"].append(msg)
                await self.broadcast_log(job, msg)
            elif return_code == 0:
                job["status"] = "completed"
                job["progress"] = 100
                job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = "[System] Job completed successfully."
                job["logs"].append(msg)
                await self.broadcast_log(job, msg)
                await self.broadcast_complete(job)
            else:
                job["status"] = "failed"
                job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = f"Process failed with return code {return_code}"
                msg = f"[Error] Process failed with return code {return_code}"
                job["logs"].append(msg)
                await self.broadcast_log(job, msg)
                await self.broadcast_error(job)
                
        except Exception as e:
            job["status"] = "failed"
            job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job["error"] = str(e)
            msg = f"[Error] Job execution crashed: {e}"
            job["logs"].append(msg)
            await self.broadcast_log(job, msg)
            await self.broadcast_error(job)
        finally:
            job["process"] = None
            await self.broadcast_status(job)
            await self.broadcast_queue_update()
            
    def _is_repetitive_match(self, last_line: str, new_line: str) -> bool:
        if "it/s]" in last_line and "it/s]" in new_line:
            return True
        if "Iteration" in last_line and "Iteration" in new_line:
            return True
        return False
        
    def parse_progress(self, line: str, job: dict):
        if "TRAINING_VIEWER_URL:" in line:
            parts = line.split("TRAINING_VIEWER_URL:")
            if len(parts) > 1:
                job["training_viewer_url"] = parts[1].strip()
                
        if "ACE-ZERO POSE ESTIMATION" in line:
            job["progress"] = 15
            job["status"] = "processing"
        if "Extracting frames" in line:
            job["progress"] = 20
        if "Extracted" in line and "frames" in line:
            job["progress"] = 30
        if "3DGS TRAINING" in line:
            job["progress"] = 40
            job["status"] = "processing"
            
        if "STEP 1:" in line: 
            job["progress"] = 10
        if "STEP 2:" in line: 
            job["progress"] = 40
        if "STEP 3:" in line: 
            job["progress"] = 70
        
        if "it/s]" in line and "/" in line:
            try:
                parts = line.split("|")
                if len(parts) >= 2:
                    step_part = parts[-1].strip()
                    if "/" in step_part:
                        current, total = step_part.split("/")[0], step_part.split("/")[1].split()[0]
                        current, total = int(current.strip()), int(total.strip())
                        train_progress = (current / total) * 55
                        job["progress"] = 40 + int(train_progress)
            except:
                pass

        if "PIPELINE COMPLETE" in line or "PIPELINE SUMMARY" in line: 
            job["progress"] = 100
            
    async def broadcast_status(self, job):
        job_id = job["job_id"]
        message = {
            "event": "job.status",
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "filename": job["filename"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
            "training_viewer_url": job.get("training_viewer_url")
        }
        await self.send_to_subscribers(job_id, message)
        
    async def broadcast_log(self, job, line):
        job_id = job["job_id"]
        message = {
            "event": "job.log",
            "job_id": job_id,
            "line": line,
            "timestamp": time.time()
        }
        await self.send_to_subscribers(job_id, message)

    async def broadcast_complete(self, job):
        job_id = job["job_id"]
        ply_files = []
        pattern = str(RESULTS_DIR / "**" / "*.ply")
        for f in glob.glob(pattern, recursive=True):
            try:
                ply_files.append(str(Path(f).relative_to(OUTPUT_DIR)))
            except:
                ply_files.append(Path(f).name)
            
        message = {
            "event": "job.complete",
            "job_id": job_id,
            "artifacts": {
                "ply": ply_files,
                "output_dir": str(OUTPUT_DIR)
            }
        }
        await self.send_to_subscribers(job_id, message)
        
    async def broadcast_error(self, job):
        job_id = job["job_id"]
        message = {
            "event": "job.error",
            "job_id": job_id,
            "error": job["error"] or "Unknown error"
        }
        await self.send_to_subscribers(job_id, message)
        
    async def broadcast_queue_update(self):
        queue_data = []
        for j_id, job in self.jobs.items():
            queue_data.append({
                "job_id": j_id,
                "status": job["status"],
                "progress": job["progress"],
                "filename": job["filename"],
                "created_at": job["created_at"]
            })
            
        dead = []
        for ws in self.global_ws_subscribers:
            try:
                await ws.send_json({
                    "event": "queue.update",
                    "queue": queue_data
                })
            except:
                dead.append(ws)
        for ws in dead:
            self.global_ws_subscribers.discard(ws)

    async def send_to_subscribers(self, job_id: str, message: dict):
        subscribers = self.ws_subscribers[job_id]
        if not subscribers:
            return
            
        dead = []
        for ws in subscribers:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
                
        for ws in dead:
            subscribers.discard(ws)


import sys
state = AppState()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing"""
    try:
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        state.current_video_path = str(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    """Training request parameters"""
    video_path: Optional[str] = None
    mode: str = "ace_zero"  # ace_zero, simple, chunked
    max_steps: int = 7000
    data_factor: int = 2
    fps: float = 1.5
    with_ut: bool = False
    with_eval3d: bool = False
    early_stopping: bool = True
    use_zero123: bool = False
    quality_mode: str = "balanced"
    min_registration_confidence: int = 1000
    depth_model: str = "depth_anything"
    skip_ace: Optional[bool] = None
    
    # Advanced Hyperparameters
    sh_degree: int = 3
    means_lr: float = 0.00016
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    ssim_lambda: float = 0.2
    random_bkgd: bool = False
    pose_opt: bool = True
    app_opt: bool = False


@app.post("/api/jobs/submit")
async def submit_job(request: TrainRequest = TrainRequest()):
    """Submit a training job to the queue"""
    video_path = request.video_path or state.current_video_path
    
    if not video_path:
        uploads_dir = UPLOADS_DIR
        if uploads_dir.exists():
            files = sorted(uploads_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            video_files = [f for f in files if f.suffix.lower() in ['.mp4', '.mov', '.avi']]
            if video_files:
                video_path = str(video_files[0])
                state.current_video_path = video_path

    if not video_path:
        raise HTTPException(
            status_code=400, 
            detail="No video uploaded. Please upload a video first."
        )
        
    config = request.dict()
    job_id = state.enqueue_job(video_path, config)
    
    # Broadcast queue update to all global WebSocket clients
    await state.broadcast_queue_update()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job successfully enqueued."
    }


@app.post("/api/train")
async def start_training(request: TrainRequest = TrainRequest()):
    """Start training pipeline (Legacy alias for submit_job)"""
    return await submit_job(request)


@app.post("/api/stop")
async def stop_training():
    """Stop current active training (Legacy alias for cancel active job)"""
    if state.active_job_id:
        success = await state.cancel_job(state.active_job_id)
        if success:
            await state.broadcast_queue_update()
            return {"message": "Active job cancelled"}
    raise HTTPException(status_code=400, detail="No active training job to stop")


@app.get("/api/status")
async def get_status():
    """Get current active training status and logs (Backward Compatibility)"""
    if state.active_job_id:
        job = state.jobs.get(state.active_job_id)
        return {
            "status": "training_running" if job["status"] in ["running", "processing"] else job["status"],
            "progress": job["progress"],
            "mode": job["config"].get("mode", "ace_zero"),
            "logs": job["logs"][-50:],
            "training_viewer_url": job.get("training_viewer_url"),
            "output_dir": str(OUTPUT_DIR)
        }
    return {
        "status": "idle",
        "progress": 0,
        "mode": "ace_zero",
        "logs": [],
        "training_viewer_url": None,
        "output_dir": str(OUTPUT_DIR)
    }


@app.get("/api/jobs/queue")
async def get_jobs_queue():
    """List all jobs in the queue"""
    queue_data = []
    for j_id, job in state.jobs.items():
        queue_data.append({
            "job_id": j_id,
            "status": job["status"],
            "progress": job["progress"],
            "filename": job["filename"],
            "created_at": job["created_at"]
        })
    return {"jobs": queue_data}


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get specific job details"""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
        "config": job["config"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        "training_viewer_url": job.get("training_viewer_url")
    }


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, offset: int = 0, limit: int = 100):
    """Get paginated logs for a specific job"""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    logs = job["logs"][offset:offset+limit]
    return {
        "logs": logs,
        "total": len(job["logs"]),
        "offset": offset,
        "limit": limit
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Cancel/delete job from queue or running"""
    success = await state.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
    await state.broadcast_queue_update()
    return {"message": f"Job {job_id} cancelled."}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List all available checkpoints (.pt files)"""
    output_base = OUTPUT_DIR
    checkpoints = []
    
    if not output_base.exists():
        return {"checkpoints": []}

    # Search for .pt files in unified output structure
    patterns = [
        str(RESULTS_DIR / "**" / "*.pt"),
        str(RESULTS_DIR / "ckpts" / "*.pt"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for f in files:
            path = Path(f)
            try:
                parts = path.parts
                chunk_name = next((p for p in parts if "result" in p.lower()), "Result")
                name = f"{chunk_name} - {path.name}"
            except:
                name = path.name
                
            if not any(c["path"] == str(path) for c in checkpoints):
                checkpoints.append({
                    "path": str(path),
                    "name": name,
                    "id": str(path.relative_to(output_base)) if path.is_relative_to(output_base) else path.name,
                    "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)
                })
    
    checkpoints.sort(key=lambda x: x["name"])
    return {"checkpoints": checkpoints}


@app.get("/api/ply")
async def list_ply_files():
    """List all available PLY files (3D models)"""
    output_base = OUTPUT_DIR
    ply_files = []
    
    if not output_base.exists():
        return {"ply_files": []}

    patterns = [
        str(output_base / "**" / "*.ply"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for f in files:
            path = Path(f)
            ply_files.append({
                "path": str(path),
                "name": path.name,
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)
            })
    
    ply_files.sort(key=lambda x: x["name"])
    return {"ply_files": ply_files}


@app.get("/api/renders")
async def list_renders():
    """List all rendered images"""
    renders = []
    
    if not OUTPUT_DIR.exists():
        return {"renders": []}

    patterns = [
        str(OUTPUT_DIR / "**" / "renders" / "*.png"),
        str(OUTPUT_DIR / "**" / "renders" / "*.jpg"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for f in files:
            path = Path(f)
            renders.append({
                "path": str(path),
                "name": path.name,
                "url": f"/api/render/{path.name}"
            })
    
    return {"renders": renders}


@app.get("/api/render/{filename}")
async def get_render(filename: str):
    """Serve a rendered image"""
    pattern = str(OUTPUT_DIR / "**" / "renders" / filename)
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        raise HTTPException(status_code=404, detail="Render not found")
    
    return FileResponse(files[0])


class ViewRequest(BaseModel):
    ckpt_paths: Optional[List[str]] = None


@app.post("/api/view")
async def start_viewer(req: ViewRequest = None):
    """Start 3D viewer with specified checkpoints"""
    if state.process_type == "viewer" and state.process:
        try:
            state.process.terminate()
        except:
            pass
        
    state.process_type = "viewer"
    
    ckpt_paths = []
    if req and req.ckpt_paths:
        ckpt_paths = req.ckpt_paths
    else:
        results_pattern = str(RESULTS_DIR / "**" / "*.pt")
        results_ckpts = glob.glob(results_pattern, recursive=True)
        if results_ckpts:
            ckpt_paths = [sorted(results_ckpts, reverse=True)[0]]
        else:
            fallback_pattern = str(OUTPUT_DIR / "**" / "*.pt")
            fallback_candidates = glob.glob(fallback_pattern, recursive=True)
            if fallback_candidates:
                ckpt_paths = [fallback_candidates[0]]

    if not ckpt_paths:
        raise HTTPException(status_code=404, detail=f"No checkpoints found in {OUTPUT_DIR}")

    viewer_script = GSPLAT_ROOT / "examples" / "simple_viewer_3dgut.py"
    port = 8092
    
    cmd = [
        get_python_executable(), str(viewer_script),
        "--port", str(port),
        "--backend", "gsplat",
        "--ckpt"
    ] + ckpt_paths
    
    print(f"Starting viewer: {cmd}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GSPLAT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    state.process = subprocess.Popen(
        cmd,
        cwd=GSPLAT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    def log_viewer_output(process):
        try:
            for line in iter(process.stdout.readline, ''):
                print(f"[Viewer] {line.strip()}", flush=True)
            process.stdout.close()
        except Exception as e:
            print(f"[Viewer] Log thread error: {e}")
            
    t = threading.Thread(target=log_viewer_output, args=(state.process,))
    t.daemon = True
    t.start()
    
    state.viewer_url = f"http://localhost:{port}"
    return {"url": state.viewer_url, "checkpoints": ckpt_paths}


@app.post("/api/viewer/stop")
async def stop_viewer():
    """Stop the 3D viewer"""
    if state.process_type == "viewer" and state.process:
        try:
            state.process.terminate()
            state.process_type = None
            return {"message": "Viewer stopped"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "No viewer running"}


@app.get("/api/info")
async def get_info():
    """Get server and pipeline information"""
    expected_ace_output = OUTPUT_DIR / "acezero_output"
    sparse_dir = expected_ace_output / "sparse" / "0"
    has_existing_poses = (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists()
    return {
        "server_version": "2.1.0",
        "pipeline_modes": {
            "ace_zero": "Level 2 ACE-Zero (AI Poses → 3DGS) - RECOMMENDED",
            "simple": "Level 1 Simple Training (pre-processed data)",
            "chunked": "Level 3 Chunked Pipeline (COLMAP-based)"
        },
        "quality_modes": ["fast", "balanced", "quality"],
        "default_mode": "ace_zero",
        "output_dir": str(OUTPUT_DIR),
        "gsplat_root": str(GSPLAT_ROOT),
        "uploads_dir": str(UPLOADS_DIR),
        "websocket_url": "ws://localhost:8081/api/ws",
        "has_existing_poses": has_existing_poses
    }



@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except:
        gpu_available = False
        gpu_name = None
        
    training_status = "idle"
    if state.active_job_id:
        active_job = state.jobs.get(state.active_job_id)
        if active_job:
            training_status = active_job["status"]
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "training_status": training_status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training queue and logs streaming.
    """
    await websocket.accept()
    state.global_ws_subscribers.add(websocket)
    
    subscribed_job_id = None
    
    try:
        # Re-send the queue status immediately on connection
        queue_data = []
        for j_id, job in state.jobs.items():
            queue_data.append({
                "job_id": j_id,
                "status": job["status"],
                "progress": job["progress"],
                "filename": job["filename"],
                "created_at": job["created_at"]
            })
            
        await websocket.send_json({
            "event": "connected",
            "queue": queue_data
        })
        
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            event = msg.get("event")
            
            if event == "subscribe":
                job_id = msg.get("job_id")
                # Unsubscribe from previous if any
                if subscribed_job_id and websocket in state.ws_subscribers[subscribed_job_id]:
                    state.ws_subscribers[subscribed_job_id].remove(websocket)
                    
                subscribed_job_id = job_id
                state.ws_subscribers[job_id].add(websocket)
                
                # Send current state & replay logs
                job = state.jobs.get(job_id)
                if job:
                    await websocket.send_json({
                        "event": "job.status",
                        "job_id": job_id,
                        "status": job["status"],
                        "progress": job["progress"],
                        "filename": job["filename"],
                        "created_at": job["created_at"],
                        "started_at": job["started_at"],
                        "completed_at": job["completed_at"],
                        "error": job["error"],
                        "training_viewer_url": job.get("training_viewer_url")
                    })
                    await websocket.send_json({
                        "event": "logs_replay",
                        "job_id": job_id,
                        "logs": job["logs"]
                    })
                    
            elif event == "unsubscribe":
                job_id = msg.get("job_id")
                if websocket in state.ws_subscribers[job_id]:
                    state.ws_subscribers[job_id].remove(websocket)
                if subscribed_job_id == job_id:
                    subscribed_job_id = None
                    
            elif event == "ping":
                await websocket.send_json({"event": "pong"})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clean up
        state.global_ws_subscribers.discard(websocket)
        if subscribed_job_id and websocket in state.ws_subscribers[subscribed_job_id]:
            state.ws_subscribers[subscribed_job_id].remove(websocket)


# =============================================================================
# STATIC FILES
# =============================================================================

# Mount static files last (catch-all)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    print("=" * 60)
    print("3DGS Demo Server v2.1 - Job Queue & WebSockets Active")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"GSplat root: {GSPLAT_ROOT}")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:8080")
    print("API docs available at http://localhost:8080/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
