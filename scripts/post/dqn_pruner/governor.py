#!/usr/bin/env python3
"""
DQN Governor (Director inference).

Builds the 32-dim state the pretrained director expects and writes cfg_governor.yml
{action, overrides}. Run by the pipeline manager as `python governor.py <result_dir>`.

State signals come from the run's SceneProfile ($HYPERSPLAT_PROFILE, written by the manager)
and fall back to reading the result/ACE directories directly when no profile is available:

    ace_confidence   registration quality of the ACE-Zero poses in [0, 1] (see
                     `ace_confidence_from_conf`), 1.0 when no poses can be found (legacy value)
    gs_psnr          latest validation PSNR of the current result dir (numeric step sort)
    gs_density       latest num_GS
    psnr_ref         first PSNR measured in this run (reference for relative scaling)
    psnr_prev        previous PSNR measurement (for the trend), or None
    psnr_best        best PSNR measured in this run so far
    density_ref      the run's own Gaussian budget: recorded CAP_MAX decision, else the
                     largest of points.count / num_GS seen so far, else 200000 (legacy)

`director_vector` turns that dict into the 32-dim tensor; train_director.py uses the same
helper so training and inference share one set of normalizers.
"""
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root on sys.path so hypersplat.pipeline.params imports when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[Governor] %(message)s')
logger = logging.getLogger(__name__)

# Legacy constants, used only when the corresponding signal is missing.
LEGACY_PSNR_SCALE = 40.0          # old vec[9] = psnr / 40
LEGACY_DENSITY_SCALE = 200000.0   # old vec[12] = num_GS / 200000
DEFAULT_MIN_REGISTRATION_CONFIDENCE = 1000.0  # manager's --min-registration-confidence default

# Relative triggers (replace density < 10000 and psnr > 30).
# 10000 / 200000 = 5% of the Gaussian budget: "model is starving for Gaussians".
LOW_DENSITY_FRACTION = 0.05
# "Converged at its best": within this many dB of the run's best PSNR and the last
# measurement moved less than this.
PLATEAU_DB = 0.1

# Director input layout (indices the pretrained model was trained with):
#   [0-7]   uncertainty / progress slot 0 (train_director writes iteration progress to [0])
#   [8]     ace_confidence              [9]  PSNR (relative, see below)
#   [12]    density (relative)          [15] low-density trigger   [16] converged trigger
#   the remaining indices are unused (zero).
IDX_PROGRESS = 0
IDX_ACE_CONF = 8
IDX_PSNR = 9
IDX_DENSITY = 12
IDX_LOW_DENSITY = 15
IDX_CONVERGED = 16


def _quantile(values: List[float], q: float) -> float:
    """Linear-interpolated quantile of a non-empty list (stdlib; q in [0, 1])."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def ace_confidence_from_conf(conf: Dict[str, float], threshold: float,
                             n_frames: Optional[int] = None) -> Optional[float]:
    """Pose-registration quality in [0, 1] from per-frame ACE-Zero confidences.

    score = registered_fraction * clamp((p10 - thr) / (median - thr), 0, 1)

    i.e. how far the weak tail (10th percentile) sits above the registration threshold,
    relative to the typical frame's margin. 1.0 = every frame registered about as well as
    the median one; 0.0 = the weakest tenth barely passed the threshold (or the median did
    not). registered_fraction = n_registered / n_frames penalizes dropped frames.
    Drone fixture: ~0.98; handheld WhatsApp night clip: ~0.78.
    """
    values = [float(c) for c in conf.values()] if conf else []
    if not values:
        return None
    median = _quantile(values, 0.5)
    p10 = _quantile(values, 0.1)
    if median <= threshold:
        tail = 0.0
    else:
        tail = min(1.0, max(0.0, (p10 - threshold) / (median - threshold)))
    frac = 1.0
    if n_frames:
        registered = sum(1 for c in values if c >= threshold)
        frac = min(1.0, registered / float(n_frames))
    return frac * tail


def director_vector(state: Dict[str, Any], progress: Optional[float] = None):
    """32-dim director input from a state dict (shared by governor and train_director).

    Scaling (values only; indices unchanged from the pretrained layout):
      vec[8]  ace_confidence (already in [0, 1]; missing -> 1.0 as before)
      vec[9]  0.5 * psnr / psnr_ref  -> 0.5 at the first measurement, >0.5 when improving.
              The old psnr/40 put a 23 dB scene at 0.58, so the mid-point keeps the value in
              the range the network saw. Without a reference: psnr / 40 (legacy).
      vec[12] num_GS / density_ref (the run's own cap) -> 1.0 when the budget is full.
              Without a reference: num_GS / 200000 (legacy).
      vec[15] 1 if num_GS < 5% of density_ref (legacy: < 10000 = 5% of 200000)
      vec[16] 1 if PSNR has converged at the run's best: psnr >= psnr_best - 0.1 dB and the
              last measurement moved < 0.1 dB and the run improved over psnr_ref
              (legacy: psnr > 30 dB, i.e. an absolute quality bar that most scenes never hit)
    """
    import torch

    vec = torch.zeros(32, dtype=torch.float32)
    if progress is not None:
        vec[IDX_PROGRESS] = float(progress)

    psnr = float(state.get("gs_psnr") or 0.0)
    density = float(state.get("gs_density") or 0.0)
    psnr_ref = state.get("psnr_ref")
    density_ref = state.get("density_ref")

    vec[IDX_ACE_CONF] = float(state.get("ace_confidence", 1.0))
    if psnr_ref and psnr_ref > 0:
        vec[IDX_PSNR] = 0.5 * psnr / float(psnr_ref)
    else:
        vec[IDX_PSNR] = psnr / LEGACY_PSNR_SCALE
    ref = float(density_ref) if density_ref and density_ref > 0 else LEGACY_DENSITY_SCALE
    vec[IDX_DENSITY] = density / ref

    vec[IDX_LOW_DENSITY] = 1.0 if density < LOW_DENSITY_FRACTION * ref else 0.0
    vec[IDX_CONVERGED] = 1.0 if psnr_converged(state) else 0.0
    return vec


def psnr_converged(state: Dict[str, Any]) -> bool:
    """Relative 'high quality' trigger: PSNR plateaued at the run's best, above its start."""
    psnr = state.get("gs_psnr")
    prev = state.get("psnr_prev")
    best = state.get("psnr_best")
    ref = state.get("psnr_ref")
    if not psnr or prev is None or best is None or ref is None:
        return False
    return (psnr >= best - PLATEAU_DB and abs(psnr - prev) < PLATEAU_DB and psnr > ref)


def _same_dir(a: Any, b: Path) -> bool:
    try:
        pa = Path(str(a))
        if pa.is_absolute():
            return pa.resolve() == b.resolve()
        # Relative paths (as in the fixtures) are matched on their tail.
        return b.resolve().as_posix().endswith(pa.as_posix())
    except (OSError, ValueError):
        return False


class PipelineGovernor:
    def __init__(self, state_dir: Path, profile=None):
        self.state_dir = Path(state_dir)
        # state_dir is the 3DGS result dir of the last training run (it holds stats/),
        # usually <OUTPUT_BASE>/results/<name>; ACE output lives at <OUTPUT_BASE>/acezero_output.
        candidates = [self.state_dir.parent.parent / "acezero_output",
                      self.state_dir.parent / "acezero_output"]
        self.ace_dir = next((c for c in candidates if c.exists()), candidates[-1])
        if profile is None:
            try:
                from hypersplat.pipeline.params import SceneProfile
                profile = SceneProfile.from_env()
            except Exception as e:  # never let the profile break the governor
                logger.warning(f"Scene profile unavailable ({e}); reading files directly.")
                profile = None
        self.profile = profile

        # Hyperparameter defaults
        self.current_action = "STEADY"
        self.overrides = {}

    @property
    def results_dir(self) -> Path:
        """Resolved on every observation: stats/ may only appear after training has run
        (train_director builds the governor before the first training)."""
        if (self.state_dir / "stats").exists():
            return self.state_dir
        legacy = self.state_dir.parent / "results" / "acezero_3dgs"
        return legacy if (legacy / "stats").exists() else self.state_dir

    # ------------------------------------------------------------------ signals
    def _pget(self, key: str, default: Any = None) -> Any:
        return self.profile.get(key, default) if self.profile is not None else default

    def _registration_threshold(self) -> float:
        dec = self._pget("decisions.MIN_REGISTRATION_CONFIDENCE")
        if isinstance(dec, dict) and dec.get("value") is not None:
            try:
                return float(dec["value"])
            except (TypeError, ValueError):
                pass
        return DEFAULT_MIN_REGISTRATION_CONFIDENCE

    def _ace_confidence(self) -> Optional[float]:
        conf = self._pget("ace.conf")
        if not conf:
            try:
                from hypersplat.pipeline.params.signals import read_ace_poses
                ace = read_ace_poses(self.ace_dir) if self.ace_dir.exists() else None
                conf = ace["conf"] if ace else None
            except Exception as e:
                logger.warning(f"Could not read ACE poses: {e}")
                conf = None
        if not conf:
            return None
        n_frames = self._pget("frames.count") or len(conf)
        return ace_confidence_from_conf(conf, self._registration_threshold(), n_frames)

    def _val_history(self) -> List[dict]:
        """All validation rows of this run in chronological order (numeric step sort)."""
        rows = self._pget("train.val") or []
        rows = [r for r in rows if isinstance(r, dict) and r.get("psnr") is not None]
        if rows:
            return rows
        try:
            from hypersplat.pipeline.params.signals import read_val_stats
            return read_val_stats(self.results_dir)
        except Exception as e:
            logger.warning(f"Could not read validation stats: {e}")
            return []

    def _current_rows(self, history: List[dict]) -> List[dict]:
        rows = [r for r in history if r.get("result_dir") is not None
                and _same_dir(r["result_dir"], self.results_dir)]
        if not rows:
            try:
                from hypersplat.pipeline.params.signals import read_val_stats
                rows = read_val_stats(self.results_dir)
            except Exception:
                rows = []
        return sorted(rows, key=lambda r: int(r.get("step", 0)))

    def _density_ref(self, history: List[dict]) -> Optional[float]:
        dec = self._pget("decisions.CAP_MAX")
        if isinstance(dec, dict) and dec.get("value"):
            try:
                return float(dec["value"])
            except (TypeError, ValueError):
                pass
        seen = [float(r["num_GS"]) for r in history if r.get("num_GS")]
        points = self._pget("points.count")
        if points:
            seen.append(float(points))
        return max(seen) if seen else None

    def observe_state(self) -> Dict[str, Any]:
        """Build the state dict from the scene profile (or the result/ACE files)."""
        state: Dict[str, Any] = {
            "ace_confidence": 1.0,
            "gs_psnr": 0.0,
            "gs_density": 0.0,
            "genvs_aggressiveness": 0.5,
            "psnr_ref": None,
            "psnr_prev": None,
            "psnr_best": None,
            "density_ref": None,
        }

        conf = self._ace_confidence()
        if conf is not None:
            state["ace_confidence"] = conf

        history = self._val_history()
        current = self._current_rows(history)
        latest = current[-1] if current else (history[-1] if history else None)
        if latest is not None:
            state["gs_psnr"] = float(latest.get("psnr", 0.0))
            state["gs_density"] = float(latest.get("num_GS", 0) or 0)
        # Chronological sequence: other dirs' rows first (profile order), then this dir's.
        timeline = [r for r in history if r not in current] + current if history else current
        psnrs = [float(r["psnr"]) for r in timeline if r.get("psnr") is not None]
        if psnrs:
            state["psnr_ref"] = psnrs[0]
            state["psnr_best"] = max(psnrs)
            state["psnr_prev"] = psnrs[-2] if len(psnrs) > 1 else None
        state["density_ref"] = self._density_ref(timeline)

        logger.info("State: " + json.dumps(state, default=str))
        return state

    def decide_action(self, state: Dict[str, Any]):
        """Decide next hyperparameters using the Neural DQN Agent."""
        try:
            # Lazy import to avoid circular dependencies
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from agent import DQNAgent

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

            vec = director_vector(state)
            nz = {i: round(float(v), 4) for i, v in enumerate(vec.tolist()) if v != 0}
            logger.info(f"Director input (non-zero indices): {nz}")

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
