#!/usr/bin/env python3
"""
DQN Governor via Rule-Based Heuristic (Stage 1)
Eventually this will be replaced by a Neural Network, but we start with 
a formal State Machine to prove the architecture.
"""
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='[Governor] %(message)s')
logger = logging.getLogger(__name__)

# Future Integration Note:
# This governor logic is designed to be decoupled from the specific trainer execution.
# For the integrated `demo_server.py`, we will need to wrap the `_map_action_director`
# calls into an API endpoint or WebSocket message handler to drive the server-side pipeline.
# The `overrides` dictionary format is compatible with the `PipelineConfig` patch method.

class PipelineGovernor:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.ace_dir = state_dir.parent / "acezero_output"
        self.results_dir = state_dir.parent / "results" / "acezero_3dgs"
        
        # Hyperparameter defaults
        self.current_action = "STEADY"
        self.overrides = {}

    def observe_state(self) -> Dict[str, float]:
        """Read all config/metric files to build the State Vector."""
        state = {
            "ace_confidence": 1.0,
            "gs_psnr": 0.0,
            "gs_density": 0.0,
            "genvs_aggressiveness": 0.5
        }
        
        # 1. Read ACE Metrics (if available)
        ace_cfg = self.ace_dir / "cfg_acezero.yml"
        # Ideally ACE would dump a stats file, but we can infer from logs or filtered poses
        # For now, let's look for the filtered poses file as a proxy for badness
        filtered_poses = self.ace_dir / "poses_final_filtered.txt"
        all_poses = self.ace_dir / "poses_final.txt"
        
        if all_poses.exists():
            total = len(all_poses.read_text().splitlines())
            if total > 0:
                filtered = 0
                if filtered_poses.exists():
                    # If filtered exists, counting lines is tricky as it only has GOOD ones
                    # logic in pipeline: if filtered exists, it REPLACED original? 
                    # Wrapper writes "poses_final_filtered.txt".
                    # Let's assume pipeline passed filtered one.
                    # We need a metric file.
                    pass
        
        # [TODO: Update ACE wrapper to dump stats.json]
        
        # 2. Read 3DGS Metrics
        # Find latest validation stats
        stats_dir = self.results_dir / "stats"
        if stats_dir.exists():
            stats_files = sorted(stats_dir.glob("val_step*.json"))
            if stats_files:
                latest = stats_files[-1]
                try:
                    with open(latest, 'r') as f:
                        data = json.load(f)
                        state["gs_psnr"] = data.get("psnr", 0.0)
                        state["gs_density"] = data.get("num_GS", 0)
                except:
                    pass

        return state

    def decide_action(self, state: Dict[str, float]):
        """Decide next hyperparameters using the Neural DQN Agent."""
        try:
            # Lazy import to avoid circular dependencies
            import torch
            from agent import DQNAgent
            from config import ActionSpace, StateSpace
            
            # 1. Load Agent
            model_path = self.state_dir.parent.parent.parent / "scripts" / "post" / "dqn_pruner" / "pretrained_pruner_director.pt"
            if not model_path.exists():
                 model_path = Path(__file__).parent / "pretrained_pruner_director.pt"
            
            # Director Agent Config
            class AgentConfig:
                INPUT_DIM = 32
                HIDDEN_DIM = 256
                ACTION_DIM = 4
                USE_GPU = False 
                
            # Construct 32-dim State Vector (Placeholder implementation for V1)
            # [0-7] Uncertainty (8-dim)
            # [8-11] Progress (4-dim)
            # [12-19] Geometric Health (8-dim)
            # [20-27] Coverage (8-dim)
            # [28-31] History (4-dim)
            
            vec = torch.zeros(32, dtype=torch.float32)
            
            # Fill metrics (Matching train_director.py)
            vec[8] = state.get("ace_confidence", 1.0) 
            vec[9] = state.get("gs_psnr", 0.0) / 40.0 
            vec[12] = state.get("gs_density", 0.0) / 200000.0
            
            # Heuristic Triggers
            vec[15] = 1.0 if state.get("gs_density", 0) < 10000 else 0.0
            vec[16] = 1.0 if state.get("gs_psnr", 0) > 30.0 else 0.0
            
            agent = DQNAgent(AgentConfig)
            
            if model_path.exists():
                try:
                    agent.load(str(model_path))
                except Exception as e:
                    logger.warning(f"Failed to load model: {e}. Using random init.")
            else:
                logger.warning("No pretrained director model found. Using random init (exploration mode).")

            # 2. Predict Action
            action_idx = agent.predict(vec).item()
            
            # 3. Map to Overrides
            self._map_action_director(action_idx)
            
        except Exception as e:
            logger.error(f"CRITICAL: DQN Inference Failed: {e}")
            # Instead of a silent fallback, we raise to ensure visibility in the 'pure RL' workflow
            raise RuntimeError(f"DQN Governor failed to produce a decision: {e}")

    def _map_action_director(self, action_idx: int):
        """Map Director Agent actions to pipeline overrides."""
        options = ["DETERMINISTIC_FILL", "RANDOM_JITTER", "PURGE_WORST_VIEW", "WAIT_FINE_TUNE"]
        action_name = options[action_idx]
        self.current_action = action_name
        
        if action_name == "DETERMINISTIC_FILL":
            self.overrides = {
                "genvs_guidance": 7.5,
                "genvs_scheduler": "ddim",
                "genvs_sample_mode": "deterministic"
            }
        elif action_name == "RANDOM_JITTER":
            self.overrides = {
                "genvs_guidance": 7.5,
                "genvs_scheduler": "ddpm",
                "genvs_sample_mode": "random",
                "genvs_noise_level": 0.5
            }
        elif action_name == "PURGE_WORST_VIEW":
            # Pipeline needs to know to run purge logic
            # We communicate this via a special flag that manager.py checks?
            # Or just set overrides that trigger pruning script?
            # Let's set a flag the manager can read.
            self.overrides = {
                "perform_purge": True,
                "prune_threshold": 0.5 # High threshold
            }
        elif action_name == "WAIT_FINE_TUNE":
            self.overrides = {
                "training_steps_add": 100,
                "skip_genvs": True
            }

    # Heuristic methods removed to enforce Pure RL decisions.

    def write_overrides(self):
        """Write the chosen action to cfg_governor.yml"""
        out_path = self.state_dir / "cfg_governor.yml"
        
        output = {
            "action": self.current_action,
            "overrides": self.overrides
        }
        
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            yaml.dump(output, f)
        
        logger.info(f"Action '{self.current_action}' saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: governor.py <state_dir>")
        sys.exit(1)
        
    state_dir = Path(sys.argv[1])
    
    gov = PipelineGovernor(state_dir)
    state = gov.observe_state()
    gov.decide_action(state)
    gov.write_overrides()
