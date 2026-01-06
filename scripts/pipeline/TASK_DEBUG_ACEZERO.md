<!-- id: task-debug-acezero -->
# Task: Debug ACE-Zero Integration Errors

## Context
The condensed `automated_intelligent_pipeline.py` failed at the ACE-Zero step.
Diagnosis revealed:
1. `conda` command not found in WSL (environment issue).
2. Found correct environment `ace0` in `~/miniconda3/envs/ace0`.
3. Wrapper logic updated to use absolute path to WSL python.
4. `iterations_max` bug (5000->10) fixed.
5. Parallel execution disabled to prevent Windows crashes.

## Status: COMPLETED
The pipeline is now running successfully in WSL using the `ace0` environment.

## Next Steps
- Monitor the training process.
- Verify 3DGS output in `results/acezero_3dgs`.
