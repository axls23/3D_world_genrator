# Enhanced Video to COLMAP with 3DGUT Quality Filtering

This enhanced version of the video-to-COLMAP pipeline incorporates quality filtering based on the 3DGUT paper principles to ensure optimal frame selection for 3D reconstruction, targeting PSNR values above 30dB.

## 🔬 Quality Filtering Features

### Based on 3DGUT Research Findings

The 3DGUT paper demonstrates that frame quality significantly impacts 3D reconstruction accuracy. Our enhanced pipeline implements:

- **Motion Blur Detection**: Laplacian variance analysis to detect and filter blurry frames
- **Feature Quality Assessment**: ORB feature counting to ensure sufficient keypoints for matching
- **PSNR Calculation**: Peak Signal-to-Noise Ratio targeting >30dB for high-quality reconstruction  
- **Contrast Analysis**: RMS contrast evaluation to ensure sufficient image detail
- **Temporal Consistency**: Frame-to-frame quality assessment for smooth reconstruction

### Quality Metrics Implementation

#### 1. Motion Blur Detection
```python
# Laplacian variance method
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
is_blurred = laplacian_var < blur_threshold  # Default: 100.0
```

#### 2. Feature Count Assessment  
```python
# ORB feature detection
orb = cv2.ORB_create(nfeatures=2000)
keypoints = orb.detect(gray, None)
sufficient_features = len(keypoints) >= feature_threshold  # Default: 500
```

#### 3. PSNR Calculation
```python
# Peak Signal-to-Noise Ratio
mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
psnr = 20 * np.log10(255.0 / np.sqrt(mse))
high_quality = psnr >= min_psnr  # Default: 30.0 dB
```

## 🚀 Enhanced Usage

### Basic Quality-Filtered Processing
```bash
python enhanced_video_to_colmap.py --video_path video.mp4 --output_dir ./dataset --min_psnr 30
```

### High-Quality Mode (Strict Filtering)
```bash
python enhanced_video_to_colmap.py     --video_path video.mp4     --output_dir ./dataset     --fps 1     --min_psnr 35     --max_frames 200
```

### Fast Mode (Relaxed Filtering)
```bash
python enhanced_video_to_colmap.py     --video_path video.mp4     --output_dir ./dataset     --fps 3     --min_psnr 25
```

## 📊 New Parameters

| Parameter | Description | Default | Purpose |
|-----------|-------------|---------|---------|
| `--min_psnr` | Minimum PSNR threshold (dB) | 30.0 | Target quality from 3DGUT |
| `--disable_quality_filtering` | Skip quality filtering | False | Use all extracted frames |
| `--max_frames` | Maximum frames to process | None | Limit processing time |

## 📁 Enhanced Output Structure

```
output_directory/
├── images/                           # Original extracted frames
├── filtered_images/                  # Quality-filtered frames ⭐
├── images_2/ (optional)             # Downsampled filtered images
├── sparse/                          # COLMAP reconstruction
├── database.db                     # COLMAP database
├── quality_analysis_report.json    # Detailed quality metrics ⭐
└── enhanced_processing_report.json # Comprehensive summary ⭐
```

## 📈 Quality Analysis Reports

### quality_analysis_report.json
Contains detailed per-frame analysis:
```json
{
  "frame_analysis": [
    {
      "frame_name": "frame_000001.jpg",
      "laplacian_variance": 245.6,
      "is_blurred": false,
      "feature_count": 847,
      "contrast": 67.3,
      "psnr": 32.4,
      "quality_score": 85,
      "is_good": true
    }
  ]
}
```

### enhanced_processing_report.json
Provides processing summary:
```json
{
  "quality_analysis": {
    "total_frames_extracted": 120,
    "frames_passed_quality": 89,
    "quality_retention_rate": 74.2,
    "failure_breakdown": {
      "motion_blur": 12,
      "insufficient_features": 8,
      "low_psnr": 7,
      "low_contrast": 4
    }
  }
}
```

## 🎯 Quality Scoring Algorithm

Each frame receives a quality score (0-100) based on:

- **Blur Check** (30%): Laplacian variance > threshold
- **Feature Count** (25%): ORB features ≥ minimum required  
- **PSNR** (30%): Peak Signal-to-Noise Ratio ≥ target
- **Contrast** (15%): RMS contrast > minimum

Frames scoring ≥70% are retained for COLMAP processing.

## 🔧 Advanced Configuration

### Custom Quality Thresholds
Modify the `FrameQualityAnalyzer` parameters:
```python
quality_analyzer = FrameQualityAnalyzer(
    min_psnr=30.0,        # Target PSNR from 3DGUT
    blur_threshold=100.0,  # Laplacian variance threshold
    feature_threshold=500  # Minimum ORB features
)
```

### Scenario-Specific Settings

#### Indoor Scenes
```bash
--min_psnr 25 --fps 2
```

#### Outdoor Scenes  
```bash
--min_psnr 30 --fps 1.5
```

#### Fast-Moving Objects
```bash
--min_psnr 35 --fps 3
```

## 🧪 Validation Against 3DGUT Metrics

The enhanced pipeline targets the quality metrics established in the 3DGUT paper:

| Metric | 3DGUT Target | Our Implementation |
|--------|--------------|-------------------|
| PSNR | >30 dB | Configurable (default 30 dB) |
| Feature Density | High | ORB feature counting |
| Motion Blur | Minimal | Laplacian variance filtering |
| Reconstruction Quality | Optimal | Quality-aware frame selection |

## 🏆 Benefits

1. **Higher 3D Reconstruction Quality**: By filtering poor-quality frames
2. **Faster COLMAP Processing**: Fewer frames to process after filtering
3. **Better Feature Matching**: Only frames with sufficient features retained
4. **Reduced Noise**: Motion blur and low-quality frames eliminated  
5. **Quantitative Assessment**: PSNR-based quality measurement
6. **Comprehensive Reporting**: Detailed analysis of frame quality

## 🔍 Troubleshooting Enhanced Features

### Very Few Frames Pass Quality Check
```
⚠️ Only 5% of frames passed quality check
```
**Solutions:**
- Lower `--min_psnr` threshold (e.g., from 30 to 25)
- Reduce feature threshold in code
- Check video quality and lighting conditions
- Use `--disable_quality_filtering` for comparison

### PSNR Calculation Issues
```
❌ PSNR calculation failed
```
**Solutions:**
- Ensure OpenCV is installed: `pip install opencv-python`
- Check frame dimensions compatibility
- Verify image loading success

### Processing Too Slow
```
⏱️ Quality analysis taking too long
```
**Solutions:**  
- Use `--max_frames` to limit processing
- Increase `--fps` for fewer initial frames
- Disable advanced metrics if not needed

## 📚 Dependencies

### Required
- `opencv-python` - Image analysis and PSNR calculation
- `numpy` - Numerical computations
- `scikit-image` (optional) - Advanced image metrics

### Installation
```bash
pip install opencv-python numpy scikit-image
```

## 🎓 Academic Reference

This implementation is based on principles from:
> Wu, Q., et al. "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting." NVIDIA Research.

The paper demonstrates that proper frame selection with quality metrics above PSNR 30dB significantly improves 3D reconstruction accuracy.

## 🔬 Research Applications

Ideal for:
- **3D Gaussian Splatting Training**: High-quality frame datasets
- **NeRF Training**: Quality-filtered input frames  
- **COLMAP Optimization**: Reduced processing time with better results
- **Computer Vision Research**: Quality-aware dataset preparation

---

**Note**: The enhanced quality filtering adds processing time but significantly improves final 3D reconstruction quality by ensuring only high-quality frames are used for structure-from-motion processing.
