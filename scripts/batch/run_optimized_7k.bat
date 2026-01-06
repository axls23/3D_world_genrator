@echo off
REM Advanced Neural Network-Inspired Optimization Training Run
REM 7000 steps with phase-based learning and convergence detection

echo ========================================
echo Advanced Optimized Training - 7K Steps
echo ========================================
echo.
echo Features enabled:
echo - Phase-based learning rate adaptation
echo - Polyak averaging convergence detection
echo - Gaussian population health monitoring
echo - Curvature-aware LR scheduling
echo.

python examples/simple_trainer.py mcmc ^
  --data_dir "C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\dataset" ^
  --data_factor 2 ^
  --result_dir "C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\RESULTS_OPTIMIZED_7K" ^
  --camera_model pinhole ^
  --save_ply ^
  --with_ut ^
  --with_eval3d ^
  --max_steps 7000 ^
  --eval_steps 3000 7000 ^
  --save_steps 3000 7000 ^
  --enable_phase_optimization ^
  --enable_polyak_averaging ^
  --enable_population_monitoring ^
  --enable_curvature_scheduling ^
  --polyak_decay 0.99 ^
  --convergence_patience 3 ^
  --min_psnr_delta 0.001 ^
  --disable_video

echo.
echo ========================================
echo Training Complete!
echo Check results in: RESULTS_OPTIMIZED_7K
echo ========================================
pause

