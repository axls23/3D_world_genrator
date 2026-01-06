# Folder Organization Plan

## Current Status: 🔴 Disorganized
Your root directory has 30+ files mixed with directories - hard to navigate!

## Proposed Structure: 🟢 Clean & Professional

```
gsplat/
├── 📁 scripts/              # All executable scripts
│   ├── pipeline/           # Pipeline scripts
│   │   ├── automated_intelligent_pipeline.py
│   │   ├── enhanced_video_to_colmap.py
│   │   ├── video_to_colmap.py
│   │   └── images_to_colmap.py
│   ├── batch/              # Batch files
│   │   ├── run_3dgut_fixed.bat
│   │   ├── run_optimized_7k.bat
│   │   └── video_to_colmap.bat
│   ├── analysis/           # Analysis scripts
│   │   ├── chart_script.py
│   │   ├── chart_script_1.py
│   │   ├── create_report.py
│   │   └── run_3dgut_test.py
│   └── utils/              # Utility scripts
│       └── script.py
│
├── 📁 docs/                # All documentation
│   ├── guides/             # User guides
│   │   ├── ENHANCED_README.md
│   │   ├── VIDEO_TO_COLMAP_README.md
│   │   ├── PARALLEL_TRAINING_GUIDE.md
│   │   └── IMPLEMENTATION_GUIDE.md
│   ├── analysis/           # Analysis reports
│   │   ├── 3DGUT_TRAINING_SUMMARY.md
│   │   ├── COMPARISON_ANALYSIS.md
│   │   └── EXPLORATION.md
│   ├── assets/             # Documentation assets
│   │   ├── pipeline_flowchart.png
│   │   └── quality_filter_chart.png
│   └── source/             # Sphinx documentation (keep as is)
│       └── ...
│
├── 📁 data/                # All data files
│   ├── VIDEO/              # Input videos
│   ├── PLAYROOM_FIXED/     # Pipeline outputs
│   ├── RESULTS/            # Training results
│   └── dataset/            # Legacy dataset
│
├── 📁 examples/            # Example scripts (keep as is)
│   ├── simple_trainer.py
│   ├── datasets/
│   ├── benchmarks/
│   └── ...
│
├── 📁 gsplat/              # Core library (keep as is)
│   ├── __init__.py
│   ├── cuda/
│   ├── strategy/
│   └── ...
│
├── 📁 tests/               # Test files (keep as is)
│   └── ...
│
├── 📁 build/               # Build artifacts (keep as is)
│   └── ...
│
├── 📁 assets/              # Project assets
│   └── test_garden.npz
│
├── 📄 README.md            # Main README
├── 📄 setup.py             # Package setup
├── 📄 LICENSE              # License
├── 📄 CITATION.bib         # Citation
├── 📄 MANIFEST.in          # Manifest
└── 📄 .gitignore           # Git ignore

```

## Organization Categories

### 1. **Scripts** → `scripts/`
- ✅ Pipeline scripts
- ✅ Batch files
- ✅ Analysis scripts
- ✅ Utility scripts

### 2. **Documentation** → `docs/`
- ✅ User guides
- ✅ Analysis reports
- ✅ Visual assets (flowcharts, charts)
- ✅ Sphinx documentation (already there)

### 3. **Data** → `data/`
- ✅ Videos
- ✅ Pipeline outputs
- ✅ Results
- ✅ Datasets

### 4. **Root Level** (Keep Clean!)
- ✅ Only essential files: README, setup.py, LICENSE, etc.
- ❌ No random scripts or docs

## Benefits

### Before:
- 30+ files in root directory
- Hard to find specific scripts
- Unclear what's documentation vs code
- Messy and unprofessional

### After:
- ~10 items in root (all essential)
- Clear categorization
- Easy navigation
- Professional structure
- GitHub-friendly

## Migration Steps

1. Create new directories
2. Move files to new locations
3. Update import paths (if needed)
4. Test functionality
5. Clean up empty directories
6. Update .gitignore
7. Commit changes

## Files to Organize

### Scripts (30 files)
```
scripts/pipeline/
  - automated_intelligent_pipeline.py
  - enhanced_video_to_colmap.py
  - video_to_colmap.py
  - images_to_colmap.py

scripts/batch/
  - run_3dgut_fixed.bat
  - run_optimized_7k.bat
  - video_to_colmap.bat

scripts/analysis/
  - chart_script.py
  - chart_script_1.py
  - create_report.py
  - run_3dgut_test.py

scripts/utils/
  - script.py
```

### Documentation (10 files)
```
docs/guides/
  - ENHANCED_README.md
  - VIDEO_TO_COLMAP_README.md
  - PARALLEL_TRAINING_GUIDE.md
  - IMPLEMENTATION_GUIDE.md

docs/analysis/
  - 3DGUT_TRAINING_SUMMARY.md
  - COMPARISON_ANALYSIS.md
  - EXPLORATION.md

docs/assets/
  - pipeline_flowchart.png
  - quality_filter_chart.png
```

### Data (Already organized)
```
data/
  VIDEO/
  PLAYROOM_FIXED/
  RESULTS/
  dataset/
```

### Keep in Root (10 files)
```
Root/
  - README.md
  - setup.py
  - LICENSE
  - CITATION.bib
  - MANIFEST.in
  - .gitignore
  - .clang-format
  - .clangd_template
  - .gitmodules
  - formatter.sh
```

## Execution

Run the organization script:
```bash
python organize_folders.py
```

Or execute manually step-by-step.


