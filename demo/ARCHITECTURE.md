System Architecture: 3DGS Reconstruction Pipeline

> **Note: everything below "Current Implementation" describes a future/aspirational architecture (separate `/api/v1` gateway, `JobManager`/`PipelineRouter`/`HealthMonitor` service classes, a React/Vue frontend on port 3000, a YAML pipeline config schema). None of that exists in the codebase today.** For the real, currently-running system, see the "Current Implementation" section immediately below and read `hypersplat/services/api/server.py` directly — it is the source of truth for routes and behavior.

## Current Implementation

The demo is a **single FastAPI process** (`hypersplat/services/api/server.py`), not a gateway + microservices layout:

- **Entry points**: `demo/demo_server.py` imports the `app` object from `hypersplat/services/api/server.py` and runs it with `uvicorn` on port `8081`; running `hypersplat/services/api/server.py` directly (its own `__main__`) binds port `8080` instead.
- **State**: a single in-process `AppState` instance (`state = AppState()`) tracks job/training state, plus an `async def worker_loop(self)` coroutine on `AppState` that pulls and executes queued work — there is no separate `JobManager`, `PipelineRouter`, or `HealthMonitor` class.
- **Frontend**: no separate React/Vue app on port 3000. The same FastAPI app serves the static HTML/JS UI directly via `app.mount("/", StaticFiles(...), name="static")`.
- **Real routes** (verify against `hypersplat/services/api/server.py` for exact signatures):

| Method | Route | Notes |
|--------|-------|-------|
| POST | `/api/upload` | Upload a video file |
| POST | `/api/jobs/submit` | Submit a new job to the queue |
| POST | `/api/train` | Start a training job (legacy/simple path) |
| POST | `/api/stop` | Stop the current training job |
| GET | `/api/status` | Get current training status/logs |
| GET | `/api/jobs/queue` | List all jobs in the queue |
| GET | `/api/jobs/{job_id}` | Get one job's details |
| GET | `/api/jobs/{job_id}/logs` | Get one job's logs |
| DELETE | `/api/jobs/{job_id}` | Cancel/delete a job |
| GET | `/api/checkpoints` | List available checkpoints |
| GET | `/api/ply` | List PLY model files |
| GET | `/api/renders` | List rendered preview images |
| GET | `/api/render/{filename}` | Fetch one rendered image |
| POST | `/api/view` | Start the 3D viewer (POST only) |
| POST | `/api/viewer/stop` | Stop the 3D viewer (separate route, not `DELETE /viewer`) |
| GET | `/api/info` | Server/environment info |
| GET | `/api/health` | Health check |
| WS | `/api/ws` | WebSocket for live status/log streaming |

There is no `/api/v1/*` namespace, no `/api/models` or `/api/assets` route, and viewer stop is its own endpoint rather than a `DELETE` on `/viewer`.

---

## Aspirational / Future Design (unimplemented)

The remainder of this document was written as a forward-looking design sketch. It does not reflect the current code and should be treated as a proposal, not a reference.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React/Vue)                              │
│                         http://localhost:3000                               │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY LAYER                                   │
│                       http://localhost:8080                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   /api/v1   │  │  /api/jobs  │  │  /api/ws    │  │  /api/assets        │ │
│  │   REST API  │  │  Job Queue  │  │  WebSocket  │  │  Static Files       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└──────────┬──────────────┬───────────────┬──────────────────┬────────────────┘
           │              │               │                  │
           ▼              ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVICE ORCHESTRATION LAYER                          │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐│
│  │   JobManager      │  │   PipelineRouter  │  │    HealthMonitor          ││
│  │   - Queue jobs    │  │   - Route to GPU  │  │    - Check GPU status     ││
│  │   - Track status  │  │   - Load balance  │  │    - Monitor memory       ││
│  │   - Retry failed  │  │   - Timeout mgmt  │  │    - Alert on failures    ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘│
└──────────┬──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE EXECUTION LAYER                            │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│   │  VIDEO INGEST   │───▶│   ACE-ZERO      │───▶│      3DGS MCMC         │ │
│   │  - Validate     │    │   (Adaptive)    │    │      Training          │ │
│   │  - Extract FPS  │    │   - Poses       │    │      - Gaussians       │ │
│   │  - Thumbnail    │    │   - Depth       │    │      - Render          │ │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│          │                       │                        │                 │
│          ▼                       ▼                        ▼                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ARTIFACT STORAGE                               │   │
│   │  - uploads/         - acezero_output/        - results/             │   │
│   │  - chunks/          - checkpoints/           - renders/             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Gateway Endpoints

### REST API v1 (/api/v1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload video file |
| POST | /train | Start training job |
| DELETE | /train | Cancel training job |
| GET | /status | Get job status |
| GET | /checkpoints | List available checkpoints |
| GET | /models | List PLY model files |
| GET | /renders | List rendered images |
| POST | /viewer | Start 3D viewer |
| DELETE | /viewer | Stop 3D viewer |

### Job Queue API (/api/jobs)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /submit | Submit new job to queue |
| GET | /{job_id} | Get job details |
| GET | /{job_id}/logs | Get job logs (paginated) |
| DELETE | /{job_id} | Cancel/delete job |
| GET | /queue | List all jobs in queue |

### WebSocket API (/api/ws)

| Event | Direction | Payload |
|-------|-----------|---------|
| `job.status` | Server→Client | `{job_id, status, progress}` |
| `job.log` | Server→Client | `{job_id, line, timestamp}` |
| `job.complete` | Server→Client | `{job_id, artifacts}` |
| `job.error` | Server→Client | `{job_id, error}` |
| `subscribe` | Client→Server | `{job_id}` |
| `unsubscribe` | Client→Server | `{job_id}` |

## Job State Machine

```
                    ┌──────────────┐
                    │   QUEUED     │
                    └──────┬───────┘
                           │ Worker picks up
                           ▼
                    ┌──────────────┐
            ┌──────▶│  RUNNING     │──────┐
            │       └──────┬───────┘      │
            │              │              │
    Retry   │              │              │ Timeout/Error
            │              ▼              │
            │       ┌──────────────┐      │
            └───────│  PROCESSING  │──────┘
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ COMPLETED│  │  FAILED  │  │ CANCELLED│
       └──────────┘  └──────────┘  └──────────┘
```

## Pipeline Stages (Detailed)

### Stage 1: Video Ingest
- Validate format (MP4, MOV, AVI, WEBM)
- Extract metadata (FPS, resolution, duration)
- Generate thumbnail for preview
- Estimate processing time

### Stage 2: ACE-Zero (Adaptive)
- Analyze video characteristics
- Auto-tune hyperparameters
- Run pose estimation
- Filter low-confidence poses
- Export point cloud

### Stage 3: 3DGS Training
- Initialize from ACE-Zero point cloud
- MCMC-based Gaussian fitting
- Progressive checkpointing
- Early stopping on convergence

### Stage 4: Post-Processing
- Export final PLY model
- Generate preview renders
- Compute quality metrics
- Cleanup temporary files

## Configuration Schema

```yaml
pipeline:
  mode: "ace_zero"  # ace_zero | simple | chunked
  
  ingest:
    max_file_size_mb: 500
    allowed_formats: [".mp4", ".mov", ".avi", ".webm"]
    fps_override: null  # Auto-detect if null
    
  ace_zero:
    quality_mode: "balanced"  # fast | balanced | quality
    min_confidence: 1000
    auto_tune: true
    
  training:
    max_steps: 30000
    batch_size: 8192
    checkpoint_interval: 1000
    early_stopping:
      enabled: true
      patience: 5
      min_delta: 0.001
      
  output:
    export_ply: true
    render_preview: true
    keep_checkpoints: [1000, 5000, 10000, "final"]
```
