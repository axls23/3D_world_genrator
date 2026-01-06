@echo off
REM Enhanced Video to COLMAP Dataset Preparation Script for Windows
REM Based on: https://github.com/jonstephens85/gsplat_3dgut
REM Enhanced with: 3DGUT quality filtering and frame optimization

echo ==============================================
echo Enhanced Video to COLMAP Dataset Preparation
echo ==============================================
echo.
echo This enhanced version includes:
echo - Motion blur detection and filtering
echo - Feature quality assessment
echo - PSNR calculation for frame quality
echo - Comprehensive quality reporting
echo.

REM Get user inputs
set /p VIDEO_PATH="Enter path to video file: "
set /p OUTPUT_DIR="Enter output directory name (default: dataset): "
set /p FPS="Enter FPS for frame extraction (default: 2): "
set /p DOWNSAMPLE="Enter downsample factor (2, 4, 8, or leave empty): "
set /p QUALITY_FILTER="Enable quality filtering? (y/n, default: y): "
set /p MIN_PSNR="Enter minimum PSNR threshold (default: 30): "
set /p MAX_FRAMES="Maximum frames to process (default: all): "

REM Set defaults
if "%FPS%"=="" set FPS=2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=dataset
if "%QUALITY_FILTER%"=="" set QUALITY_FILTER=y
if "%MIN_PSNR%"=="" set MIN_PSNR=30

echo.
echo Processing with the following settings:
echo Video: %VIDEO_PATH%
echo Output: %OUTPUT_DIR%
echo FPS: %FPS%
echo Downsample: %DOWNSAMPLE%
echo Quality filtering: %QUALITY_FILTER%
echo Min PSNR: %MIN_PSNR%
echo Max frames: %MAX_FRAMES%
echo.

REM Activate conda environment and run enhanced Python script
echo Activating conda environment and running enhanced processing...
if "%QUALITY_FILTER%"=="n" (
    REM Use basic script without quality filtering
    if "%DOWNSAMPLE%"=="" (
        conda run -n 3dgrut python video_to_colmap.py --video_path "%VIDEO_PATH%" --output_dir "%OUTPUT_DIR%" --fps %FPS%
    ) else (
        conda run -n 3dgrut python video_to_colmap.py --video_path "%VIDEO_PATH%" --output_dir "%OUTPUT_DIR%" --fps %FPS% --downsample %DOWNSAMPLE%
    )
) else (
    REM Use enhanced script with quality filtering
    set "ENHANCED_CMD=conda run -n 3dgrut python enhanced_video_to_colmap.py --video_path "%VIDEO_PATH%" --output_dir "%OUTPUT_DIR%" --fps %FPS%"

    if not "%DOWNSAMPLE%"=="" set "ENHANCED_CMD=%ENHANCED_CMD% --downsample %DOWNSAMPLE%"
    if not "%MIN_PSNR%"=="30" set "ENHANCED_CMD=%ENHANCED_CMD% --min_psnr %MIN_PSNR%"
    if not "%MAX_FRAMES%"=="" set "ENHANCED_CMD=%ENHANCED_CMD% --max_frames %MAX_FRAMES%"

    echo Running: %ENHANCED_CMD%
    %ENHANCED_CMD%
)

echo.
if errorlevel 1 (
    echo Processing failed! Check the error messages above.
    echo.
    echo Troubleshooting tips:
    echo 1. Ensure FFmpeg and COLMAP are installed and in PATH
    echo 2. Check video file exists and is readable
    echo 3. Verify sufficient disk space for output
    echo 4. For quality filtering issues, try lowering --min_psnr
) else (
    echo.
    echo ✅ Enhanced processing completed successfully!
    echo.
    echo Dataset is ready at: %OUTPUT_DIR%
    echo.
    echo Output includes:
    echo - Quality analysis report
    echo - Frame quality statistics
    echo - Comprehensive processing summary
    echo.
    echo Use this dataset for 3D Gaussian Splatting training!
)

echo.
pause