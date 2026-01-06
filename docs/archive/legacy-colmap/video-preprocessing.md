# Video to COLMAP Dataset Preparation

This guide helps you convert video files into COLMAP-ready datasets for 3D Gaussian Splatting training, based on the [gsplat_3dgut](https://github.com/jonstephens85/gsplat_3dgut) workflow.

## 📋 Prerequisites

Before running the scripts, ensure you have the following software installed:

### Required Software
1. **COLMAP** - Download from [COLMAP releases](https://github.com/colmap/colmap/releases)
   - Install and ensure `colmap` command is available in your PATH
   
2. **FFmpeg** - Download from [FFmpeg website](https://ffmpeg.org/download.html)
   - Install and ensure `ffmpeg` command is available in your PATH
   
3. **ImageMagick** (Optional, for downsampling)
   - Download from [ImageMagick website](https://imagemagick.org/script/download.php#windows)
   - Install and ensure `magick` command is available in your PATH

### Python Environment
The scripts use your `3dgrut` conda environment which should already have:
- `pycolmap`
- `numpy`
- `pathlib` (built-in)

## 🚀 Usage

### Method 1: Python Script (Recommended)

```bash
# Basic usage
conda run -n 3dgrut python video_to_colmap.py --video_path ./data/VIDEO/WhatsApp\ Video\ 2025-09-11\ at\ 13.12.04_05de4ce1.mp4 --output_dir ./data/PLAYROOM

# With custom settings
conda run -n 3dgrut python video_to_colmap.py \
    --video_path ./data/VIDEO/WhatsApp\ Video\ 2025-09-11\ at\ 13.12.04_05de4ce1.mp4 \
    --output_dir ./data/PLAYROOM \
    --fps 1.5 \
    --downsample 2 \
    --camera_model SIMPLE_PINHOLE
```

### Method 2: Windows Batch Script

Double-click `video_to_colmap.bat` and follow the prompts.

## ⚙️ Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--video_path` | Path to input video file | Required | Any video format supported by FFmpeg |
| `--output_dir` | Output directory name | Required | Any valid directory path |
| `--fps` | Frames per second to extract | 2.0 | Any positive number |
| `--downsample` | Image downsampling factor | None | 2, 4, 8 |
| `--camera_model` | COLMAP camera model | SIMPLE_PINHOLE | SIMPLE_PINHOLE, PINHOLE, OPENCV_FISHEYE |

## 📁 Output Structure

The script creates the following directory structure:

```
output_directory/
├── images/                    # Original extracted frames
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── images_2/                  # Downsampled images (if --downsample 2)
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── sparse/                    # COLMAP reconstruction
│   └── 0/
│       ├── cameras.bin/.txt   # Camera parameters
│       ├── images.bin/.txt    # Image poses
│       └── points3D.bin/.txt  # 3D point cloud
├── database.db               # COLMAP feature database
└── processing_report.json    # Processing summary
```

## 🎯 Best Practices

### Video Selection
- Use videos with good lighting and minimal motion blur
- Avoid rapid camera movements
- Include overlapping views of the scene (60-80% overlap recommended)
- Ensure the subject is visible from multiple angles

### Frame Rate Selection
- **High detail scenes**: Use 1-2 FPS
- **Simple scenes**: Use 2-3 FPS
- **Fast movement**: Use higher FPS (3-5)

### Downsampling Guidelines
- **High VRAM (>8GB)**: No downsampling needed
- **Medium VRAM (4-8GB)**: Use `--downsample 2`
- **Low VRAM (<4GB)**: Use `--downsample 4` or `--downsample 8`
- Keep longest image dimension around 2000-4000 pixels

### Camera Model Selection
- **Standard cameras**: `SIMPLE_PINHOLE` (recommended)
- **Calibrated cameras**: `PINHOLE`
- **Fisheye lenses**: `OPENCV_FISHEYE`

## 🐛 Troubleshooting

### Common Issues

#### FFmpeg not found
```
Error: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH
```
**Solution**: Install FFmpeg and add it to your system PATH.

#### COLMAP not found
```
Error: COLMAP not found. Please install COLMAP and ensure it's in your PATH
```
**Solution**: Install COLMAP and add it to your system PATH.

#### No frames extracted
```
Error: No frames were extracted from the video
```
**Solutions**: 
- Check if the video file is valid
- Try a different video format
- Reduce the FPS value

#### COLMAP reconstruction failed
```
Error: COLMAP mapper failed
```
**Solutions**:
- Try with downsampled images
- Reduce the number of input images
- Check if images have sufficient overlap
- Try a different camera model

#### Out of memory during processing
**Solutions**:
- Use `--downsample 2` or higher
- Reduce the FPS to extract fewer frames
- Close other applications to free up RAM

### Quality Tips

1. **Check image quality**: Examine extracted frames in the `images/` folder
2. **Verify COLMAP results**: Check if the sparse reconstruction looks reasonable
3. **Monitor processing logs**: Watch for warnings during COLMAP steps
4. **Test with small datasets**: Start with short videos (10-30 seconds)

## 📊 Performance Expectations

| Video Length | Image Count (2 FPS) | Processing Time* |
|--------------|-------------------|------------------|
| 30 seconds   | ~60 images       | 5-10 minutes    |
| 1 minute     | ~120 images      | 10-20 minutes   |
| 2 minutes    | ~240 images      | 20-40 minutes   |
| 5 minutes    | ~600 images      | 1-2 hours       |

*Times are approximate and depend on image resolution, hardware, and scene complexity.

## 🔗 Next Steps

After successful processing, you can use the dataset for 3D Gaussian Splatting training:

```bash
# Train with 3DGUT enabled
cd examples
conda run -n 3dgrut python simple_trainer.py \
    --data_dir ../path/to/your/dataset \
    --with_ut \
    --with_eval3d
```

## 📚 Additional Resources

- [COLMAP Documentation](https://colmap.github.io/)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [gsplat_3dgut Repository](https://github.com/jonstephens85/gsplat_3dgut)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

## ⚠️ Notes

- Processing can be CPU-intensive, especially feature matching and reconstruction
- Large videos may require significant disk space for extracted frames
- The script automatically converts COLMAP output to text format for compatibility
- Processing times scale significantly with the number of input images

Happy 3D reconstruction! 🎉