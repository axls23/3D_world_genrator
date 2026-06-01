#!/usr/bin/env python3
"""
DQN 'Director' Training Script 
Learns the optimal policy for iterative 3DGS-GeNVS refinement.

Algorithm: Deep Q-Learning (DQN) with Replay Buffer
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
import json
import time
import shutil
import traceback
import os
from pathlib import Path
from collections import deque
from typing import Tuple, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "examples"))

from scripts.post.dqn_pruner.agent import DQNAgent, QNetwork
from scripts.post.dqn_pruner.governor import PipelineGovernor
from scripts.post.dqn_pruner.config import ActionSpace
from hypersplat.pipeline.manager import IntelligentPipeline, PipelineConfig

# === TRAINING CONFIG ===
EPISODES = 5
STEPS_PER_EPISODE = 20  # Max iterations per scene
BATCH_SIZE = 32
GAMMA = 0.99           # Discount factor
EPSILON_START = 1.0    # Exploration rate
EPSILON_END = 0.1
EPSILON_DECAY = 0.95
LEARNING_RATE = 1e-4
MEMORY_SIZE = 1000

# === REWARD CONFIG ===
REWARD_PSNR_SCALE = 100.0  # +PSNR * 100
COST_DETERMINISTIC = -0.5
COST_RANDOM = -5.0
COST_PURGE = -1.0
COST_WAIT = -0.1
PENALTY_ARTIFACTS = 0.5    # Per 10k floaters

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DirectorTrainer:
    def __init__(self, scene_path: str):
        self.scene_path = Path(scene_path)
        self.output_base = PROJECT_ROOT / "demo" / "output_training"
        
        # Initialize Pipeline
        class Args:
             output_dir = str(self.output_base)
             data_factor = 1
             fps = 2
             fps = 2
             # Dynamic max_steps: Each Director Step adds +500 steps
             self.base_max_steps = 1000
             max_steps = self.base_max_steps 
             save_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
             eval_steps = [] # Disable expensive video rendering per step
             init_scale = 1.0
             opacity_reg = 0.0
             scale_reg = 0.0
             app_opt = False
             sh_degree = 0 # simple training
             means_lr = 1.6e-4
             ssim_lambda = 0.2
             random_bkgd = False
             with_ut = False
             with_eval3d = False
             skip_ace = False
             colmap_input = None
             quality_mode = 'fast'
             min_confidence = 0.2
             depth_model = 'depth_anything'
             cloud_sync = False
             scene_name = "training_scene"
             streaming = False
             prune = True
             unified_stream = False
             
        self.pipeline_config = PipelineConfig(Args)
        self.pipeline_config.create_directories()
        self.pipeline = IntelligentPipeline(self.pipeline_config)
        self.governor = PipelineGovernor(self.output_base / "results" / "acezero_3dgs")
        
        # Agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using the same architecture as governor.py
        class AgentConfig:
            INPUT_DIM = 32
            HIDDEN_DIM = 256
            ACTION_DIM = 4
            USE_GPU = True
            
        self.agent = DQNAgent(AgentConfig)
        self.optimizer = optim.Adam(self.agent.q_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
        self.output_model_path = Path(__file__).parent / "pretrained_pruner_director.pt"
        
        # Runtime State
        self.current_iter = 0
        self.current_data_dir = None
        self.current_result_dir = None

        # === PERSISTENT VISER VIEWER ===
        import viser
        from gsplat_viewer import GsplatViewer
        import threading
        
        print(f"[Director] Initializing persistent viewer at http://0.0.0.0:8081")
        self.server = viser.ViserServer(port=8081)
        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            output_dir=self.output_base / "results" / "acezero_3dgs",
            mode="rendering"
        )
        
        # --- Director Dashboard UI ---
        with self.server.gui.add_folder("Director Dashboard"):
            self.ui_episode = self.server.gui.add_number("Episode", initial_value=0, disabled=True)
            self.ui_step = self.server.gui.add_number("Step", initial_value=0, disabled=True)
            self.ui_reward = self.server.gui.add_number("Total Reward", initial_value=0.0, disabled=True)
            self.ui_psnr = self.server.gui.add_number("Quality (PSNR)", initial_value=0.0, disabled=True)
            self.ui_action = self.server.gui.add_text("Last Action", initial_value="IDLE", disabled=True)
            self.ui_status = self.server.gui.add_markdown("Status: **Initializing Director**")

        self.splats = nn.ParameterDict()
        self.render_cfg = type('obj', (object,), {'sh_degree': 0})
        self.last_ckpt_loaded = None
        self.last_ckpt_mtime = 0
        
        # Start background loader
        self.loader_thread = threading.Thread(target=self._viewer_loader, daemon=True)
        self.loader_thread.start()

    def reset_viewer(self):
        """Clears the viewer state for a new episode."""
        self.splats = nn.ParameterDict()
        self.last_ckpt_loaded = None
        self.last_ckpt_mtime = 0
        # Force a black render
        self.viewer.rerender(None)
        self.ui_status.content = "Status: **Resetting Environment...**"
        print("[Viewer] Environment Reset.")

    def _viewer_loader(self):
        """Polls for new checkpoints and loads them into memory."""
        ckpt_dir = self.output_base / "results" / "acezero_3dgs" / "ckpts"
        while True:
            try:
                if ckpt_dir.exists():
                    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                    if ckpts:
                        latest = ckpts[-1]
                        latest_mtime = latest.stat().st_mtime
                        
                        # Reload if filename OR modification time changed
                        if (latest != self.last_ckpt_loaded) or (latest_mtime > self.last_ckpt_mtime):
                            # Load splats
                            ckpt = torch.load(latest, map_location=self.device, weights_only=False)
                            splat_data = ckpt["splats"]
                            N = splat_data["means"].shape[0]
                            
                            # Re-init ParameterDict if size changed
                            if not self.splats or self.splats["means"].shape[0] != N:
                                self.splats = nn.ParameterDict({
                                    k: nn.Parameter(v.to(self.device)) for k, v in splat_data.items()
                                })
                            else:
                                self.splats.load_state_dict(splat_data)
                            
                            # Infer SH Degree
                            if "shN" in splat_data:
                                n_coeffs = splat_data["shN"].shape[1] + 1
                                self.render_cfg.sh_degree = int(np.sqrt(n_coeffs) - 1)
                            else:
                                self.render_cfg.sh_degree = 0
                                
                            self.last_ckpt_loaded = latest
                            self.last_ckpt_mtime = latest_mtime
                            self.viewer.rerender(None)
                            print(f"[Viewer] Live Update: {latest.name} ({N} splats)")

                # Update Dashboard from state.json
                state_path = self.output_base / "results" / "acezero_3dgs" / "state.json"
                if state_path.exists():
                    try:
                        with open(state_path, "r") as f:
                            sdata = json.load(f)
                            self.ui_episode.value = sdata.get("episode", 0)
                            self.ui_step.value = sdata.get("step", 0)
                            self.ui_reward.value = round(sdata.get("total_reward", 0.0), 2)
                            self.ui_action.value = sdata.get("action", "N/A")
                            self.ui_psnr.value = round(sdata.get("psnr", 0.0), 2)
                    except:
                        pass
            except Exception as e:
                pass # Silent during training interruptions
            time.sleep(1)

    def _viewer_render_fn(self, camera_state, render_tab_state):
        """Standard high-quality renderer for the persistent viewer."""
        from gsplat.rendering import rasterization
        
        # Correct dimension extraction from render_tab_state
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height

        if not self.splats or "means" not in self.splats:
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Use camera-to-world and invert to get viewmat
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        viewmat = torch.linalg.inv(c2w)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)
        
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        
        if "sh0" in self.splats:
             colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        else:
             colors = torch.sigmoid(self.splats["colors"])

        RENDER_MODE_MAP = {"rgb": "RGB", "depth(accumulated)": "D", "depth(expected)": "ED", "alpha": "RGB"}

        render_colors, _, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmat[None], Ks=K[None], width=width, height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.render_cfg.sh_degree),
            near_plane=render_tab_state.near_plane, far_plane=render_tab_state.far_plane,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )
        
        # Update UI counts
        render_tab_state.total_gs_count = len(means)
        if "radii" in info:
            render_tab_state.rendered_gs_count = (info["radii"] > 0).sum().item()
        
        canvas = render_colors[0, ..., 0:3].clamp(0, 1)
        return (canvas.detach().cpu().numpy() * 255).astype(np.uint8)

    def compute_reward(self, state_t, state_t1, action):
        """Calculate reward based on PSNR gain and Compute Cost."""
        psnr_t = state_t["gs_psnr"]
        psnr_t1 = state_t1["gs_psnr"]
        
        # 1. Fidelity Gain (Reward improvements)
        fidelity = (psnr_t1 - psnr_t) * REWARD_PSNR_SCALE
        
        # 2. Compute Cost (Penalize slow/expensive actions)
        costs = [COST_DETERMINISTIC, COST_RANDOM, COST_PURGE, COST_WAIT]
        cost = costs[action]
        
        # 3. Artifact Penalty (Penalize density bloat without PSNR gain)
        density_t = state_t["gs_density"]
        density_t1 = state_t1["gs_density"]
        
        artifact_penalty = 0.0
        # If density grows by >20% but PSNR doesn't improve by >0.1
        if (density_t1 > density_t * 1.2) and (psnr_t1 < psnr_t + 0.1):
             # Use normalized penalty per 10k floaters
             bloat_count = (density_t1 - density_t) / 10000.0
             artifact_penalty = -PENALTY_ARTIFACTS * max(1.0, bloat_count)
             
        reward = fidelity + cost + artifact_penalty
        return reward

    def select_action(self, state_vec):
        """Epsilon-Greedy Action Selection"""
        if random.random() < self.epsilon:
            return random.randint(0, 3) # Random action
        else:
            return self.agent.predict(state_vec).item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        state_batch = torch.stack(states).to(self.device).squeeze(1) # [B, 32]
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device).float()
        next_state_batch = torch.stack(next_states).to(self.device).squeeze(1)
        done_batch = torch.tensor(dones).to(self.device).float()
        
        # Q(s, a)
        q_values = self.agent.q_net(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Target Q(s', a')
        with torch.no_grad():
            next_q_values = self.agent.q_net(next_state_batch)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward_batch + GAMMA * next_q_value * (1 - done_batch)
            
        loss = self.criterion(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def dict_to_tensor(self, state_dict):
        """Convert state dictionary to 32-dim tensor with heuristic features."""
        vec = torch.zeros(32, dtype=torch.float32)
        
        # Progress (Normalized)
        vec[0] = float(self.current_iter) / STEPS_PER_EPISODE
        
        # Geometric Metrics
        vec[8] = state_dict.get("ace_confidence", 1.0) 
        vec[9] = state_dict.get("gs_psnr", 0.0) / 40.0 
        vec[12] = state_dict.get("gs_density", 0.0) / 200000.0
        
        # Heuristic Triggers (One-hot style)
        vec[15] = 1.0 if state_dict.get("gs_density", 0) < 10000 else 0.0   # Low density signal
        vec[16] = 1.0 if state_dict.get("gs_psnr", 0) > 30.0 else 0.0      # High quality signal
        
        return vec.unsqueeze(0) # [1, 32]

    def run_environment_step(self, action_idx):
        """
        Interacts with the real pipeline!
        1. Write action to cfg_governor.yml
        2. Execute Pipeline Refinement
        """
        # Map action to overrides using governor logic
        self.governor._map_action_director(action_idx)
        self.governor.write_overrides()
        
        print(f"[Run] Executing Action {action_idx}: {self.governor.current_action}")
        print(f"      Overrides: {self.governor.overrides}")
        
        iter_dir = self.output_base / f"iter_{self.current_iter}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # 0. Identify Latest Checkpoint to Resume From
        resume_ckpt = None
        if self.current_result_dir:
             ckpt_dir = self.current_result_dir / "ckpts"
             if ckpt_dir.exists():
                 ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                 if ckpts:
                     resume_ckpt = ckpts[-1]
        
        # Increment max_steps for the next run (Continuous Training)
        # e.g., Step 0: 1000, Step 1: 1500, Step 2: 2000
        additional_steps = 500
        new_max_steps = self.pipeline_config.TRAINING_MAX_STEPS + additional_steps
        
        # Hack: Modify the config object in-place for the next run
        self.pipeline_config.TRAINING_MAX_STEPS = new_max_steps
        
        # Also need to add new save points
        current_save_max = self.pipeline_config.TRAINING_SAVE_STEPS[-1]
        new_saves = [current_save_max + 100 * i for i in range(1, 6)]
        self.pipeline_config.TRAINING_SAVE_STEPS.extend(new_saves)
        
        print(f"[Continuous] Resuming from {resume_ckpt.name if resume_ckpt else 'Scratch'} -> Target Steps: {new_max_steps}")

        # Execute Real Pipeline Step
        try:
            # 1. GeNVS Refinement
            self.ui_status.content = f"Status: **GeNVS Refinement (Action: {self.governor.current_action})**"
            augmented_dir = self.pipeline._perform_genvs_refinement(
                self.current_data_dir,
                self.current_result_dir,
                iter_dir,
                self.governor.overrides
            )
            
            # 2. Retraining
            # Apply overrides temporarily
            self.ui_status.content = f"Status: **Fine-tuning 3DGS...**"
            original_cfg = self.pipeline._patch_config(self.governor.overrides)
            try:
                # Pass resume_ckpt to manager
                self.current_result_dir = self.pipeline._run_training(augmented_dir, resume_ckpt=resume_ckpt)
                self.current_data_dir = augmented_dir
            finally:
                self.pipeline._restore_config(original_cfg)
                
            self.current_iter += 1
            
        except Exception as e:
            print(f"Pipeline Execution Failed: {e}")
            traceback.print_exc()
            # Return current state (penalty will be applied for no change)
            return self.governor.observe_state()
        
        return self.governor.observe_state()

    def train(self):
        print(f"Starting Training on {self.scene_path}")
        
        # 1. Run Pose Estimation Once (Global Cache)
        # Check if we already have ACE output in the training folder
        # Note: perception.py wrapper outputs to output_dir/acezero_output
        # But inside that, it seems to create *another* acezero_output based on your listing.
        # Let's check the nested path specifically found in list_dir.
        
        cached_ace_root = self.output_base / "acezero_output"
        nested_ace_output = cached_ace_root / "acezero_output" 
        
        # If the NESTED file exists, we are good.
        if nested_ace_output.exists() and (nested_ace_output / "poses_final.txt").exists():
            print(f"Found cached ACE-Zero output at {nested_ace_output}. Reusing...")
            # Ideally the pipeline expects the PARENT folder (which contains images/, sparse/, acezero_output/)
            # So we pass cached_ace_root
            ace_out = cached_ace_root 
        else:
            print(f"Cache miss at {nested_ace_output}. Running ACE-Zero Pose Estimation...")
            ace_out = self.pipeline._run_pose_estimation(self.scene_path)
            
        for episode in range(EPISODES):
            print(f"\n--- Episode {episode+1}/{EPISODES} ---")
            
            # Reset Environment (Full Reset)
            # Reset Environment (Full Reset)
            self.current_iter = 0
            
            # Explicitly reset the viewer to handle transition
            self.reset_viewer()

            # Initialize Governor State
            self.governor.current_action = "RESETTING"
            self.governor.overrides = {}
            self.governor.write_overrides()
            
            print("Resetting Scene (Running initial Training)...")
            
            # Train initial 3DGS from scratch using cached poses
            # _run_training will create a new results folder if we handle naming, 
            # but current manager overwrites 'acezero_3dgs'. 
            # For training, we might want 'episode_X_3dgs'.
            # For now, let it overwrite to save space, RL cares about the step delta.
            initial_result = self.pipeline._run_training(ace_out)
            
            self.current_data_dir = ace_out
            self.current_result_dir = initial_result
            
            # Observe Initial State
            state_dict = self.governor.observe_state()
            state_vec = self.dict_to_tensor(state_dict)
            
            total_reward = 0
            
            
            for step in range(STEPS_PER_EPISODE):
                # Select Action
                action = self.select_action(state_vec)
                
                # Execute
                state_dict_next = self.run_environment_step(action)
                state_vec_next = self.dict_to_tensor(state_dict_next)
                
                # Reward
                reward = self.compute_reward(state_dict, state_dict_next, action)
                total_reward += reward
                
                # Store
                done = (step == STEPS_PER_EPISODE - 1)
                self.memory.push(state_vec, action, reward, state_vec_next, done)
                
                # Train
                loss = self.train_step()
                
                print(f"  Step {step}: Act={action}, R={reward:.2f}, Loss={loss:.4f}, Epsilon={self.epsilon:.2f}")
                
                # Save state for dashboard
                with open(self.output_base / "results" / "acezero_3dgs" / "state.json", "w") as f:
                    json.dump({
                        "episode": episode + 1, 
                        "step": step, 
                        "total_reward": total_reward,
                        "action": self.governor.current_action,
                        "psnr": state_dict.get("gs_psnr", 0.0)
                    }, f)
                
                state_dict = state_dict_next
                state_vec = state_vec_next
                
                if done:
                    break
            
            # Decary Epsilon
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
            
            # Save Model
            torch.save(self.agent.q_net.state_dict(), self.output_model_path)
            print(f"Episode Reward: {total_reward:.2f}. Model saved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train_director.py <video_path>")
        sys.exit(1)
        
    trainer = DirectorTrainer(sys.argv[1])
    trainer.train()
