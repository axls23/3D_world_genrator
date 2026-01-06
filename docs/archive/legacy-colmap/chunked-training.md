# Parallel Multi-Chunk Training Guide

## Overview
The `automated_intelligent_pipeline.py` now supports **true parallel training** of multiple video chunks simultaneously across multiple threads/GPUs.

## Key Features

### 1. **Multi-GPU Support**
- Automatically detects all available GPUs
- Distributes chunks across GPUs (1 chunk per GPU by default)
- Example: 4 chunks + 2 GPUs = 2 chunks training simultaneously

### 2. **Single-GPU Multi-Threading**
- Can train multiple chunks on a single GPU (if memory permits)
- Useful for smaller models or high VRAM GPUs
- Default: Sequential (1 at a time), can enable parallel with flag

### 3. **Thread-Safe Execution**
- Each training process runs in isolated environment
- Unique `CUDA_VISIBLE_DEVICES` per process
- Unique `MASTER_PORT` to avoid conflicts
- Progress tracking for all parallel jobs

## Usage Examples

### Basic Usage (Auto-Detect)
```bash
python automated_intelligent_pipeline.py --video_path "data/VIDEO/your_video.mp4"
```
- Automatically detects GPUs
- Multi-GPU: Trains multiple chunks in parallel
- Single-GPU: Trains sequentially (safe default)

### Force Parallel on Single GPU
```bash
python automated_intelligent_pipeline.py \
  --video_path "data/VIDEO/your_video.mp4" \
  --force_parallel \
  --single_gpu_parallel 2
```
- Trains 2 chunks simultaneously on single GPU
- Requires sufficient VRAM (8GB+ recommended)

### Specify Max Parallel Workers
```bash
python automated_intelligent_pipeline.py \
  --video_path "data/VIDEO/your_video.mp4" \
  --max_parallel 4
```
- Manually set to 4 parallel workers
- Useful if you have 4+ GPUs

### Full Custom Configuration
```bash
python automated_intelligent_pipeline.py \
  --video_path "data/VIDEO/your_video.mp4" \
  --training_steps 10000 \
  --max_parallel 3 \
  --force_parallel \
  --single_gpu_parallel 3 \
  --min_psnr 38.0
```

## Configuration Options

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video_path` | str | **Required** | Path to input video |
| `--training_steps` | int | 7000 | Max training steps per chunk |
| `--min_psnr` | float | 35.0 | Minimum PSNR for quality filtering |
| `--max_parallel` | int | Auto | Max parallel workers (overrides auto-detect) |
| `--force_parallel` | flag | False | Enable parallel on single GPU |
| `--single_gpu_parallel` | int | 2 | Max parallel jobs on single GPU |

### Code Configuration (PipelineConfig)

```python
class PipelineConfig:
    # Parallel processing
    MAX_PARALLEL_WORKERS = None  # Auto-detect
    FORCE_SINGLE_GPU_PARALLEL = False  # Enable single-GPU parallel
    SINGLE_GPU_MAX_PARALLEL = 2  # Max jobs on single GPU
```

## Parallelization Strategy

### Multi-GPU (Recommended)
```
System: 2 GPUs, 4 Chunks
┌─────────────────────────────────┐
│ GPU 0: Chunk 0 → Chunk 2       │
│ GPU 1: Chunk 1 → Chunk 3       │
└─────────────────────────────────┘
Result: 2x speedup
```

### Single-GPU Parallel (Advanced)
```
System: 1 GPU (12GB VRAM), 4 Chunks, --force_parallel
┌─────────────────────────────────┐
│ GPU 0: Chunk 0 + Chunk 1       │
│        (then)                   │
│        Chunk 2 + Chunk 3       │
└─────────────────────────────────┘
Result: ~1.5-1.8x speedup (if VRAM allows)
```

## Performance Tips

### 1. **GPU Memory Management**
- Monitor VRAM usage: `nvidia-smi -l 1`
- Single-GPU parallel needs **6-8GB VRAM per chunk**
- Reduce `--training_steps` if OOM errors occur

### 2. **Optimal Worker Count**
- **Multi-GPU**: Set `max_parallel = number of GPUs`
- **Single-GPU**: Start with 2, increase if stable
- **CPU**: Limited by memory bandwidth, not recommended for >4 parallel

### 3. **Chunk Size Optimization**
- Smaller chunks = faster individual training
- More chunks = better parallelization
- Balance: 5-18 second chunks (auto-adjusted by pipeline)

### 4. **Progressive Scaling**
```bash
# Test with 1 worker first
python automated_intelligent_pipeline.py --video_path video.mp4 --max_parallel 1

# Then scale up
python automated_intelligent_pipeline.py --video_path video.mp4 --max_parallel 2

# Full parallel (if stable)
python automated_intelligent_pipeline.py --video_path video.mp4 --force_parallel
```

## Output Format

### Console Output
```
Parallel Training Configuration:
  Total chunks to train: 3
  Available GPUs: 2 (IDs: [0, 1])
  Max parallel workers: 2
  Strategy: Multi-GPU
======================================================================

[Thread 0 | GPU 0] Training chunk 0...
[Thread 1 | GPU 1] Training chunk 1...

[Progress] 1/3 chunks completed
[Thread 2 | GPU 0] Training chunk 2...

[Progress] 2/3 chunks completed
[Progress] 3/3 chunks completed

======================================================================
Parallel Training Complete!
  Total time: 156.3s
  Successful: 3/3
  Average time per chunk: 78.2s
  Speedup: 1.94x
======================================================================
```

### Results Structure
```
data/PLAYROOM_FIXED/
├── chunks/                    # Video chunks
│   ├── chunk_000.mp4
│   ├── chunk_001.mp4
│   └── chunk_002.mp4
├── datasets/                  # COLMAP datasets
│   ├── dataset_chunk_000/
│   ├── dataset_chunk_001/
│   └── dataset_chunk_002/
└── results/                   # Training results (parallel)
    ├── result_chunk_000/     # Trained on GPU 0, Thread 0
    ├── result_chunk_001/     # Trained on GPU 1, Thread 1
    └── result_chunk_002/     # Trained on GPU 0, Thread 2
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce parallel workers
```bash
python automated_intelligent_pipeline.py --video_path video.mp4 --max_parallel 1
```

### Issue: "Port already in use"
**Solution**: Pipeline auto-assigns unique ports (29500 + thread_id)
- Thread 0: Port 29500
- Thread 1: Port 29501
- Thread 2: Port 29502

### Issue: All chunks train sequentially
**Check**: 
1. Do you have multiple GPUs? Run `nvidia-smi --list-gpus`
2. Single GPU? Add `--force_parallel` flag
3. Set `--max_parallel 2` explicitly

### Issue: Inconsistent speedup
**Causes**:
- GPU memory bandwidth saturation
- CPU bottleneck in data loading
- Disk I/O limitations
**Solution**: Reduce parallel workers or upgrade hardware

## Advanced: Programmatic Usage

```python
from pathlib import Path
from automated_intelligent_pipeline import IntelligentPipeline, PipelineConfig

# Create custom config
config = PipelineConfig()
config.TRAINING_MAX_STEPS = 10000
config.MAX_PARALLEL_WORKERS = 4
config.FORCE_SINGLE_GPU_PARALLEL = True
config.SINGLE_GPU_MAX_PARALLEL = 3

# Run pipeline
pipeline = IntelligentPipeline(config)
report = pipeline.run("data/VIDEO/my_video.mp4")

# Access results
print(f"Success rate: {report['success_rate']:.1f}%")
print(f"Models trained: {report['training']['models_trained']}")
```

## Benchmarks (Example)

### Sequential (1 GPU, no parallel)
- 3 chunks × 50s each = **150s total**
- Speedup: 1.0x

### Parallel (1 GPU, forced parallel, 2 workers)
- 3 chunks, 2 at a time = **~90s total**
- Speedup: 1.67x

### Multi-GPU (2 GPUs, 2 workers)
- 3 chunks, 2 parallel = **~75s total**
- Speedup: 2.0x

### Multi-GPU (4 GPUs, 4 workers)
- 4 chunks, all parallel = **~50s total**
- Speedup: 4.0x (linear scaling!)

## Best Practices

1. **Start Conservative**: Begin with auto-detect, then enable parallel
2. **Monitor Resources**: Watch GPU memory and temperature
3. **Scale Gradually**: Increase workers one at a time
4. **Use Multi-GPU**: Best speedup with minimal risk
5. **Optimize Chunks**: Intelligent chunking already optimizes for parallelization

## Next Steps

After parallel training completes:
1. View results in `data/PLAYROOM_FIXED/results/`
2. Compare point clouds: `result_chunk_XXX/ply/point_cloud_999.ply`
3. Check render quality: `result_chunk_XXX/renders/`
4. Review stats: `result_chunk_XXX/stats/`
5. Merge chunks (future feature) for complete scene reconstruction

