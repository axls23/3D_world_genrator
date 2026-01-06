"""
3DGS Demo Server with Level 2 ACE-Zero Pipeline

Features:
- Video upload and processing
- Level 2 ACE-Zero pipeline (AI poses → 3DGS)
- Real-time training status and logs
- Checkpoint management and viewing
- Interactive 3D viewer integration
- WebSocket for real-time updates

API Endpoints:
  POST /api/upload        - Upload video file
  POST /api/train         - Start training pipeline
  GET  /api/status        - Get training status
  POST /api/stop          - Stop training
  GET  /api/checkpoints   - List available checkpoints
  GET  /api/ply           - List PLY files
  POST /api/view          - Start 3D viewer
  WS   /api/ws            - Real-time updates (WebSocket)
"""

import os
import shutil
import subprocess
import threading
import asyncio
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
import glob
from typing import List, Optional
from pydantic import BaseModel

# Suppress noisy /api/status polling logs
import logging

class StatusEndpointFilter(logging.Filter):
    """Filter out repetitive /api/status log entries."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()

# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(StatusEndpointFilter())

app = FastAPI(
    title="3DGS Demo Server",
    description="Video to 3D Gaussian Splatting with ACE-Zero Pipeline",
    version="2.0.0"
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
GSPLAT_ROOT = CURRENT_DIR.parent
SCRIPTS_DIR = GSPLAT_ROOT / "scripts" / "pipeline"

# Single unified output directory
OUTPUT_DIR = CURRENT_DIR / "output"
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

from pipeline_wrapper import TrainingManager, PipelineMode


class AppState:
    """Application state management"""
    def __init__(self):
        self.training_manager = TrainingManager(GSPLAT_ROOT, OUTPUT_DIR)
        self.viewer_process = None
        self.status = "idle"  # idle, viewing, error
        self.viewer_url = None
        self.current_video_path = None
        self.process_type = None  # "viewer" or None
        self.process = None
        self.logs = []
        # WebSocket subscribers for real-time updates
        self.ws_subscribers: List[WebSocket] = []
    
    async def broadcast(self, message: dict):
        """Broadcast message to all WebSocket subscribers."""
        dead = []
        for ws in self.ws_subscribers:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        for ws in dead:
            self.ws_subscribers.remove(ws)

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
    mode: str = "ace_zero"  # ace_zero (default), simple, chunked
    max_steps: int = 7000
    data_factor: int = 2
    fps: float = 1.5
    with_ut: bool = True
    with_eval3d: bool = True
    early_stopping: bool = True  # Stop when loss converges
    use_zero123: bool = False  # Use Zero123 for novel view synthesis
    # === NEW ===
    quality_mode: str = "balanced"  # fast | balanced | quality
    min_confidence: int = 1000  # Minimum pose confidence threshold


@app.post("/api/train")
async def start_training(request: TrainRequest = TrainRequest()):
    """
    Start 3DGS training pipeline.
    
    Modes:
    - ace_zero: Level 2 ACE-Zero pipeline (default, recommended)
    - simple: Level 1 simple training on pre-processed data
    - chunked: Level 3 full chunked pipeline with COLMAP
    """
    video_path = request.video_path or state.current_video_path
    
    # Fallback: Check uploads folder for most recent file
    if not video_path:
        uploads_dir = UPLOADS_DIR
        if uploads_dir.exists():
            files = sorted(uploads_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            video_files = [f for f in files if f.suffix.lower() in ['.mp4', '.mov', '.avi']]
            if video_files:
                video_path = str(video_files[0])
                state.current_video_path = video_path
                # logger.info(f"Auto-selected text upload: {video_path}")

    if not video_path:
        raise HTTPException(
            status_code=400, 
            detail="No video uploaded. Please upload a video first."
        )

    # Validate mode
    valid_modes = [PipelineMode.LEVEL_1_SIMPLE, PipelineMode.LEVEL_2_ACE_ZERO, PipelineMode.LEVEL_3_CHUNKED]
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Valid modes: {valid_modes}"
        )

    try:
        # Configure training
        config_override = {
            "max_steps": request.max_steps,
            "data_factor": request.data_factor,
            "fps": request.fps,
            "with_ut": request.with_ut,
            "with_eval3d": request.with_eval3d,
            "early_stopping": request.early_stopping,
            "use_zero123": request.use_zero123,
            # === NEW ===
            "quality_mode": request.quality_mode,
            "min_confidence": request.min_confidence,
        }
        
        state.training_manager.start_training(
            video_path, 
            mode=request.mode,
            config_override=config_override
        )
        
        return {
            "message": "Training started",
            "mode": request.mode,
            "video": video_path,
            "config": config_override
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop")
async def stop_training():
    """Stop current training"""
    try:
        state.training_manager.stop_training()
        return {"message": "Training stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get current training status and logs"""
    status_data = state.training_manager.get_status()
    
    response = status_data.copy()
    
    # Add viewer status if active
    if state.status == "viewing":
        response["app_status"] = "viewing"
        response["viewer_url"] = state.viewer_url
    
    # Add output directory info
    response["output_dir"] = str(OUTPUT_DIR)
    
    return response


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
                
            # Avoid duplicates
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
    output_base = OUTPUT_DIR
    renders = []
    
    if not output_base.exists():
        return {"renders": []}

    patterns = [
        str(output_base / "**" / "renders" / "*.png"),
        str(output_base / "**" / "renders" / "*.jpg"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for f in files:
            path = Path(f)
            renders.append({
                "path": str(path),
                "name": path.name,
                "url": f"/api/render/{path.name}"  # Serve via API
            })
    
    return {"renders": renders}


@app.get("/api/render/{filename}")
async def get_render(filename: str):
    """Serve a rendered image"""
    # Search for the file
    pattern = str(OUTPUT_DIR / "**" / "renders" / filename)
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        raise HTTPException(status_code=404, detail="Render not found")
    
    return FileResponse(files[0])


class ViewRequest(BaseModel):
    """Viewer request parameters"""
    ckpt_paths: Optional[List[str]] = None


@app.post("/api/view")
async def start_viewer(req: ViewRequest = None):
    """Start 3D viewer with specified checkpoints"""
    # Kill existing viewer if any
    if state.process_type == "viewer" and state.process:
        try:
            state.process.terminate()
        except:
            pass
        
    state.status = "viewing"
    state.process_type = "viewer"
    
    ckpt_paths = []
    
    # Use provided paths if any
    if req and req.ckpt_paths:
        ckpt_paths = req.ckpt_paths
    else:
        # Auto-find checkpoints in unified output structure
        
        # Try results directory first (main training output)
        results_pattern = str(RESULTS_DIR / "**" / "*.pt")
        results_ckpts = glob.glob(results_pattern, recursive=True)
        if results_ckpts:
            ckpt_paths = [sorted(results_ckpts, reverse=True)[0]]
        else:
            # Fallback to any .pt in output dir
            fallback_pattern = str(OUTPUT_DIR / "**" / "*.pt")
            fallback_candidates = glob.glob(fallback_pattern, recursive=True)
            if fallback_candidates:
                ckpt_paths = [fallback_candidates[0]]

    if not ckpt_paths:
        state.status = "error"
        error_msg = f"No checkpoints found in {OUTPUT_DIR}"
        state.logs.append(error_msg)
        raise HTTPException(status_code=404, detail=error_msg)

    viewer_script = GSPLAT_ROOT / "examples" / "simple_viewer_3dgut.py"
    port = 8092
    
    cmd = [
        "python", str(viewer_script),
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
        env=env
    )
    
    state.viewer_url = f"http://localhost:{port}"
    return {"url": state.viewer_url, "checkpoints": ckpt_paths}


@app.post("/api/viewer/stop")
async def stop_viewer():
    """Stop the 3D viewer"""
    if state.process_type == "viewer" and state.process:
        try:
            state.process.terminate()
            state.status = "idle"
            state.process_type = None
            return {"message": "Viewer stopped"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "No viewer running"}


@app.get("/api/info")
async def get_info():
    """Get server and pipeline information"""
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
        "websocket_url": "ws://localhost:8080/api/ws"
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
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "training_status": state.training_manager.status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training updates.
    
    Client can send:
      {"action": "ping"}              - Keep-alive
      {"action": "get_logs", "offset": 0, "limit": 100}  - Get logs
    
    Server sends:
      {"event": "connected"}          - On connection
      {"event": "status", "data": {...}}  - Status updates
      {"event": "log", "line": "..."}     - New log line
      {"event": "heartbeat"}          - Keep-alive response
    """
    await websocket.accept()
    state.ws_subscribers.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "event": "connected",
            "data": state.training_manager.get_status()
        })
        
        # Listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = json.loads(data)
                
                if msg.get("action") == "ping":
                    await websocket.send_json({"event": "pong"})
                
                elif msg.get("action") == "get_logs":
                    offset = msg.get("offset", 0)
                    limit = msg.get("limit", 100)
                    logs = state.training_manager.logs[offset:offset + limit]
                    await websocket.send_json({
                        "event": "logs",
                        "logs": logs,
                        "offset": offset,
                        "total": len(state.training_manager.logs)
                    })
                
                elif msg.get("action") == "get_status":
                    await websocket.send_json({
                        "event": "status",
                        "data": state.training_manager.get_status()
                    })
            
            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                await websocket.send_json({"event": "heartbeat"})
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in state.ws_subscribers:
            state.ws_subscribers.remove(websocket)


# =============================================================================
# STATIC FILES
# =============================================================================

# Mount static files last (catch-all)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    print("=" * 60)
    print("3DGS Demo Server v2.0 - Level 2 ACE-Zero Pipeline")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"GSplat root: {GSPLAT_ROOT}")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:8080")
    print("API docs available at http://localhost:8080/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8080)
