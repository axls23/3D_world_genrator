# Demo Server

Web interface for video-to-3DGS training pipeline.

## Quick Start

```bash
cd demo
python demo_server.py
# Open http://localhost:8081
```

Note: `demo/demo_server.py` imports the FastAPI `app` from `hypersplat/services/api/server.py` and serves it on port `8081`. If you instead run `hypersplat/services/api/server.py` directly (its own `__main__` block), it binds port `8080`.

## Features

- Video upload and processing
- Real-time training progress/logs
- Checkpoint and PLY file management
- Interactive 3D viewer

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload video file |
| `/api/train` | POST | Start training pipeline |
| `/api/status` | GET | Get training status/logs |
| `/api/stop` | POST | Stop training |
| `/api/checkpoints` | GET | List checkpoint files |
| `/api/ply` | GET | List PLY models |
| `/api/view` | POST | Start 3D viewer |
| `/api/viewer/stop` | POST | Stop viewer |

## Training Modes

```json
{
  "mode": "ace_zero",  // Default, recommended
  "max_steps": 7000,
  "data_factor": 2,
  "fps": 1.5
}
```

| Mode | Description |
|------|-------------|
| `ace_zero` | ACE-Zero poses → 3DGS (default) |
| `simple` | Direct training on existing data |
| `chunked` | Legacy COLMAP-based pipeline |

## Output Structure

All outputs go to `demo/output/`:

```
demo/output/
├── uploads/    # Uploaded videos
├── results/    # Training results
├── acezero/    # Pose estimation
└── chunks/     # Chunked pipeline data
```

## Example Usage

### Upload and Train

```bash
# Upload video
curl -X POST -F "file=@video.mp4" http://localhost:8081/api/upload

# Start training
curl -X POST -H "Content-Type: application/json" \
  -d '{"mode":"ace_zero","max_steps":7000}' \
  http://localhost:8081/api/train

# Check status
curl http://localhost:8081/api/status
```

### View Results

```bash
# List checkpoints
curl http://localhost:8081/api/checkpoints

# Start viewer with latest checkpoint
curl -X POST http://localhost:8081/api/view
```
