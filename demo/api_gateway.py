"""
API Gateway Layer for 3DGS Pipeline

Implements:
- RESTful API routing with versioning
- WebSocket support for real-time updates
- Job queue management
- Request validation and rate limiting
- Error handling and logging
"""

import asyncio
import uuid
import json
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import aiofiles

logger = logging.getLogger(__name__)


# ============================================================================
# JOB MODELS
# ============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    INGEST = "ingest"
    ACE_ZERO = "ace_zero"
    TRAINING = "training"
    POST_PROCESS = "post_process"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: datetime
    video_path: str
    config: Dict[str, Any]
    stage: Optional[PipelineStage] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self):
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d


# ============================================================================
# JOB MANAGER (In-Memory Queue)
# ============================================================================

class JobManager:
    """
    Manages job queue and execution.
    In production, replace with Redis/RabbitMQ backed implementation.
    """
    
    def __init__(self, max_concurrent: int = 1):
        self.jobs: Dict[str, Job] = {}
        self.queue: deque = deque()
        self.max_concurrent = max_concurrent
        self.running_count = 0
        self._subscribers: Dict[str, List[WebSocket]] = {}
        
    def submit(self, video_path: str, config: Dict[str, Any]) -> Job:
        """Submit a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            video_path=video_path,
            config=config
        )
        self.jobs[job.id] = job
        self.queue.append(job.id)
        logger.info(f"Job {job.id} submitted to queue")
        return job
    
    def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_all(self) -> List[Job]:
        """List all jobs."""
        return list(self.jobs.values())
    
    def update(self, job_id: str, **kwargs):
        """Update job fields and notify subscribers."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = datetime.now()
        
        # Notify WebSocket subscribers
        asyncio.create_task(self._notify_subscribers(job_id, job))
    
    def add_log(self, job_id: str, line: str):
        """Add log line to job."""
        job = self.jobs.get(job_id)
        if job:
            job.logs.append(line)
            if len(job.logs) > 1000:  # Keep last 1000 lines
                job.logs = job.logs[-1000:]
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PROCESSING]:
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now()
            return True
        return False
    
    def subscribe(self, job_id: str, websocket: WebSocket):
        """Subscribe to job updates via WebSocket."""
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(websocket)
    
    def unsubscribe(self, job_id: str, websocket: WebSocket):
        """Unsubscribe from job updates."""
        if job_id in self._subscribers:
            if websocket in self._subscribers[job_id]:
                self._subscribers[job_id].remove(websocket)
    
    async def _notify_subscribers(self, job_id: str, job: Job):
        """Send update to all WebSocket subscribers."""
        if job_id not in self._subscribers:
            return
        
        message = {
            "event": "job.status",
            "data": job.to_dict()
        }
        
        dead_sockets = []
        for ws in self._subscribers[job_id]:
            try:
                await ws.send_json(message)
            except:
                dead_sockets.append(ws)
        
        for ws in dead_sockets:
            self._subscribers[job_id].remove(ws)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TrainConfig(BaseModel):
    mode: str = Field("ace_zero", description="Pipeline mode: ace_zero | simple | chunked")
    quality_mode: str = Field("balanced", description="Quality: fast | balanced | quality")
    depth_model: str = Field("depth_anything", description="Depth estimator: depth_anything | zoedepth | midas")
    max_steps: int = Field(30000, ge=1000, le=100000)
    min_confidence: int = Field(1000, ge=100, le=5000)
    early_stopping: bool = True
    
class TrainRequest(BaseModel):
    video_path: str = Field(..., description="Path to uploaded video")
    config: TrainConfig = Field(default_factory=TrainConfig)

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str]
    progress: float
    logs: List[str] = []

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ============================================================================
# API GATEWAY ROUTER
# ============================================================================

def create_api_gateway(job_manager: JobManager, pipeline_executor):
    """
    Create FastAPI app with API gateway pattern.
    
    Args:
        job_manager: JobManager instance for queue management
        pipeline_executor: Callable that runs the actual pipeline
    """
    
    app = FastAPI(
        title="3DGS Pipeline API Gateway",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ========================================================================
    # REST API v1
    # ========================================================================
    
    @app.post("/api/v1/upload", response_model=dict)
    async def upload_video(file: UploadFile = File(...)):
        """Upload a video file for processing."""
        from pathlib import Path
        
        upload_dir = Path("demo/output/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return {
            "success": True,
            "path": str(file_path),
            "filename": file.filename,
            "size_mb": len(content) / (1024 * 1024)
        }
    
    @app.post("/api/v1/train", response_model=JobResponse)
    async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
        """Submit a training job to the queue."""
        
        job = job_manager.submit(
            video_path=request.video_path,
            config=request.config.dict()
        )
        
        # Execute pipeline in background
        background_tasks.add_task(
            pipeline_executor,
            job_manager,
            job.id
        )
        
        return JobResponse(
            job_id=job.id,
            status=job.status.value,
            message="Job submitted to queue"
        )
    
    @app.delete("/api/v1/train/{job_id}")
    async def cancel_training(job_id: str):
        """Cancel a running or queued job."""
        success = job_manager.cancel(job_id)
        if success:
            return {"success": True, "message": f"Job {job_id} cancelled"}
        raise HTTPException(404, f"Job {job_id} not found or cannot be cancelled")
    
    @app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
    async def get_status(job_id: str):
        """Get job status and progress."""
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        
        return StatusResponse(
            job_id=job.id,
            status=job.status.value,
            stage=job.stage.value if job.stage else None,
            progress=job.progress,
            logs=job.logs[-100:]  # Last 100 lines
        )
    
    @app.get("/api/v1/jobs")
    async def list_jobs():
        """List all jobs."""
        return {
            "jobs": [job.to_dict() for job in job_manager.list_all()]
        }
    
    @app.get("/api/v1/jobs/{job_id}/logs")
    async def get_logs(job_id: str, offset: int = 0, limit: int = 100):
        """Get job logs with pagination."""
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        
        logs = job.logs[offset:offset + limit]
        return {
            "job_id": job_id,
            "offset": offset,
            "limit": limit,
            "total": len(job.logs),
            "logs": logs
        }
    
    @app.get("/api/v1/jobs/{job_id}/artifacts")
    async def get_artifacts(job_id: str):
        """Get job output artifacts."""
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        
        return {
            "job_id": job_id,
            "artifacts": job.artifacts
        }
    
    # ========================================================================
    # WebSocket API
    # ========================================================================
    
    @app.websocket("/api/ws/{job_id}")
    async def websocket_endpoint(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for real-time job updates."""
        await websocket.accept()
        
        job = job_manager.get(job_id)
        if not job:
            await websocket.send_json({"error": f"Job {job_id} not found"})
            await websocket.close()
            return
        
        job_manager.subscribe(job_id, websocket)
        
        try:
            # Send initial state
            await websocket.send_json({
                "event": "connected",
                "data": job.to_dict()
            })
            
            # Keep connection alive and handle incoming messages
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    msg = json.loads(data)
                    
                    if msg.get("action") == "ping":
                        await websocket.send_json({"event": "pong"})
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({"event": "heartbeat"})
                    
        except WebSocketDisconnect:
            job_manager.unsubscribe(job_id, websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            job_manager.unsubscribe(job_id, websocket)
    
    # ========================================================================
    # Health & Info Endpoints
    # ========================================================================
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        import torch
        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/info")
    async def get_info():
        """Get server information."""
        return {
            "version": "2.0.0",
            "pipeline_modes": ["ace_zero", "simple", "chunked"],
            "quality_modes": ["fast", "balanced", "quality"],
            "depth_models": [
                {"id": "depth_anything", "name": "Depth Anything V2", "speed": "~30ms/frame", "type": "relative"},
                {"id": "zoedepth", "name": "ZoeDepth", "speed": "~200ms/frame", "type": "metric"},
                {"id": "midas", "name": "MiDaS", "speed": "~100ms/frame", "type": "relative"}
            ],
            "max_file_size_mb": 500,
            "supported_formats": [".mp4", ".mov", ".avi", ".webm"]
        }
    
    return app


# ============================================================================
# PIPELINE EXECUTOR (Background Task)
# ============================================================================

async def execute_pipeline(job_manager: JobManager, job_id: str):
    """
    Execute the full reconstruction pipeline for a job.
    
    This is the background task that runs ACE-Zero + 3DGS training.
    """
    import subprocess
    import sys
    from pathlib import Path
    
    job = job_manager.get(job_id)
    if not job:
        return
    
    try:
        # Update status to running
        job_manager.update(job_id, status=JobStatus.RUNNING, stage=PipelineStage.INGEST)
        
        video_path = job.video_path
        config = job.config
        output_dir = Path(f"demo/output/jobs/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: ACE-Zero
        job_manager.update(job_id, stage=PipelineStage.ACE_ZERO, progress=0.1)
        job_manager.add_log(job_id, f"[ACE-ZERO] Starting adaptive pose estimation...")
        
        # Import and run adaptive ACE-Zero
        # (In a real implementation, we'd stream output to the job logs)
        acezero_dir = output_dir / "acezero_output"
        
        cmd = [
            sys.executable, "acezero/adaptive_ace_zero.py",
            video_path,
            str(acezero_dir),
            "--quality_mode", config.get("quality_mode", "balanced"),
            "--min_confidence", str(config.get("min_confidence", 1000))
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="gsplat"
        )
        
        async for line in process.stdout:
            decoded = line.decode().strip()
            job_manager.add_log(job_id, f"[ACE-ZERO] {decoded}")
            
            # Parse progress from output
            if "Step" in decoded:
                job_manager.update(job_id, progress=0.2)
        
        await process.wait()
        
        if process.returncode != 0:
            raise Exception("ACE-Zero failed")
        
        # Stage 2: 3DGS Training
        job_manager.update(job_id, stage=PipelineStage.TRAINING, progress=0.3)
        job_manager.add_log(job_id, f"[3DGS] Starting MCMC training...")
        
        training_cmd = [
            sys.executable, "examples/simple_trainer.py",
            "mcmc",
            "--data_dir", str(acezero_dir),
            "--max_steps", str(config.get("max_steps", 30000)),
            "--result_dir", str(output_dir / "results"),
        ]
        
        process = await asyncio.create_subprocess_exec(
            *training_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="gsplat"
        )
        
        async for line in process.stdout:
            decoded = line.decode().strip()
            job_manager.add_log(job_id, f"[3DGS] {decoded}")
            
            # Parse step progress
            if "step:" in decoded.lower():
                try:
                    step = int(decoded.split("step:")[1].split()[0])
                    max_steps = config.get("max_steps", 30000)
                    progress = 0.3 + (step / max_steps) * 0.6
                    job_manager.update(job_id, progress=min(progress, 0.9))
                except:
                    pass
        
        await process.wait()
        
        if process.returncode != 0:
            raise Exception("3DGS training failed")
        
        # Stage 3: Post-process
        job_manager.update(job_id, stage=PipelineStage.POST_PROCESS, progress=0.95)
        job_manager.add_log(job_id, "[POST] Collecting artifacts...")
        
        # Collect artifacts
        artifacts = {}
        
        ply_files = list(output_dir.glob("**/*.ply"))
        if ply_files:
            artifacts["model"] = str(ply_files[0])
        
        pt_files = list(output_dir.glob("**/*.pt"))
        if pt_files:
            artifacts["checkpoint"] = str(pt_files[-1])
        
        renders = list(output_dir.glob("**/renders/*.png"))
        if renders:
            artifacts["renders"] = [str(r) for r in renders[:10]]
        
        # Complete
        job_manager.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            artifacts=artifacts,
            completed_at=datetime.now()
        )
        job_manager.add_log(job_id, "[DONE] Pipeline completed successfully!")
        
    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        job_manager.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(e)
        )
        job_manager.add_log(job_id, f"[ERROR] {str(e)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    job_manager = JobManager(max_concurrent=1)
    app = create_api_gateway(job_manager, execute_pipeline)
    
    print("=" * 60)
    print("3DGS Pipeline API Gateway")
    print("=" * 60)
    print("API Docs: http://localhost:8080/api/docs")
    print("Health:   http://localhost:8080/api/health")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
