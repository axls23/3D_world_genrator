@echo off
REM Fixed 3DGUT Training Script - Disables trajectory rendering for small datasets
REM This avoids the reshape error when there are too few camera poses

cd c:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat

set PYTHONPATH=c:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat

echo ================================================================================
echo 3DGUT Training with Fixed Configuration
echo ================================================================================
echo.
echo Dataset: TEST_OUTPUT_RUN2 (Quality-filtered frames)
echo Features: 3DGUT (with_ut + with_eval3d)
echo Strategy: MCMC
echo Max Steps: 500
echo.

conda run -n 3dgrut python examples/simple_trainer.py mcmc --data_dir  --data_factor 4 --result_dir --with_ut --with_eval3d  --eval_steps  --save_ply  

echo.
if errorlevel 1 (
    echo Training encountered an error. Check output above.
) else (
    echo ================================================================================
    echo Training completed successfully!
    echo ================================================================================
    echo.
    echo Results saved to: RESULTS_3DGUT_FIXED
    echo.
    dir RESULTS_3DGUT_FIXED\ckpts
    echo.
    dir RESULTS_3DGUT_FIXED\stats
)

pause









