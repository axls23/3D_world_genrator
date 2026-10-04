# Documentation

gsplat project documentation.

## Quick Links

| Guide | Description |
|-------|-------------|
| [Pipeline Guide](getting-started/pipeline.md) | ACE-Zero → 3DGS workflow |
| [Demo Server](getting-started/demo-server.md) | Web interface usage |
| [3DGUT Features](getting-started/3dgut.md) | Distortion & rolling shutter |
| [Streaming](getting-started/streaming.md) | Video / live camera → 3DGS, online Difix, Arrow Flight |

## Reference

| Doc | Description |
|-----|-------------|
| [Development](reference/DEV.md) | Dev setup, testing, docs build |
| [GeNVS-Lite](reference/genvs.md) | Novel view synthesis |
| [DQN Pruner](reference/pruner.md) | Floater removal |

## Structure

```
docs/
├── getting-started/    # User guides
├── reference/          # Technical docs
├── archive/            # Historical/legacy docs
├── source/             # Sphinx documentation
└── assets/             # Images
```

## Building Sphinx Docs

```bash
pip install -r docs/requirements.txt
sphinx-build docs/source _build
```

## Archive

Legacy and historical documents are in `archive/`:
- COLMAP-based pipeline guides
- Technical debt analysis
- Development reports
