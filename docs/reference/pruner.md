# DQN Floater Pruner

Reinforcement learning-based post-processing to remove floater Gaussians.

## Overview

Trained DQN agent identifies and removes noisy Gaussians that appear as floaters in rendered views.

## Usage

```bash
python scripts/pipeline/dqn_floater_pruner.py \
    --ply_path results/final.ply \
    --output_path results/pruned.ply
```

## Results

Typical performance:
- **Floaters removed**: 15-25%
- **Quality preserved**: Visual quality maintained
- **Processing time**: < 1 minute

### Example

| Metric | Before | After |
|--------|--------|-------|
| Gaussians | 141,036 | 114,051 |
| Floaters removed | - | 26,985 (19.1%) |

## How It Works

1. **Feature Extraction**: Per-Gaussian attributes (opacity, scale, position)
2. **DQN Evaluation**: Agent scores each Gaussian
3. **Thresholding**: Low-score Gaussians classified as floaters
4. **Pruning**: Floaters removed from PLY file

## When to Use

- Post-training cleanup
- Before physics simulation
- For cleaner renders
