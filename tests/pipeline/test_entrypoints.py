"""Entry points (wrapper.py / server.py) must not inject defaults.

Any flag an entry point always passes becomes "user-set" in the manager and disables that
param's dynamic strategy, so unset values must be omitted from the command.
"""

import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

from hypersplat.pipeline import wrapper
from hypersplat.pipeline.wrapper import (
    TrainingManager,
    build_pipeline_args,
    parse_training_loss,
    parse_training_step,
    validate_pipeline_config,
)

AUTO_FLAGS = (
    "--fps", "--max_steps", "--data_factor", "--opacity_reg", "--scale_reg", "--sh_degree",
    "--means_lr", "--ssim_lambda", "--quality-mode", "--min-registration-confidence",
    "--depth-model", "--genvs-views", "--early-stop-patience", "--early-stop-min-delta",
    "--early-stop-min-steps", "--cap-max", "--init_scale", "--pose-opt", "--no-pose-opt",
    "--early-stopping", "--no-early-stopping", "--min_chunk_duration",
)


def _flag_value(cmd, flag):
    assert cmd.count(flag) == 1, f"{flag} appears {cmd.count(flag)} times in {cmd}"
    return cmd[cmd.index(flag) + 1]


@pytest.fixture
def manager(tmp_path):
    return TrainingManager(root_dir=tmp_path, output_dir=tmp_path / "out")


@pytest.fixture(scope="module")
def server():
    pytest.importorskip("fastapi")
    # server.py creates demo/output/* at import time; keep the tree clean.
    with mock.patch("os.makedirs"):
        sys.modules.pop("hypersplat.services.api.server", None)
        mod = importlib.import_module("hypersplat.services.api.server")
    return mod


# --- shared builder --------------------------------------------------------------------------

def test_empty_config_emits_no_auto_flags():
    assert build_pipeline_args({}) == []


def test_explicit_values_are_passed_through():
    cmd = build_pipeline_args({
        "fps": 3.0, "max_steps": 300, "data_factor": 1, "opacity_reg": 0.0, "scale_reg": 0.0,
        "depth_model": "zoedepth", "cap_max": 500000, "pose_opt": False, "early_stopping": True,
        "early_stop_patience": 100, "genvs": True, "genvs_views": 8,
    })
    assert _flag_value(cmd, "--fps") == "3.0"
    assert _flag_value(cmd, "--max_steps") == "300"
    assert _flag_value(cmd, "--data_factor") == "1"
    assert _flag_value(cmd, "--opacity_reg") == "0.0"
    assert _flag_value(cmd, "--scale_reg") == "0.0"
    assert _flag_value(cmd, "--depth-model") == "zoedepth"
    assert _flag_value(cmd, "--cap-max") == "500000"
    assert _flag_value(cmd, "--early-stop-patience") == "100"
    assert _flag_value(cmd, "--genvs-views") == "8"
    assert "--no-pose-opt" in cmd and "--pose-opt" not in cmd
    assert "--early-stopping" in cmd and "--genvs" in cmd
    assert "--early-stop-min-delta" not in cmd


def test_legacy_wrapper_early_stop_keys_and_disable():
    cmd = build_pipeline_args({"loss_patience": 50, "loss_threshold": 0.01, "min_steps": 10})
    assert _flag_value(cmd, "--early-stop-patience") == "50"
    assert _flag_value(cmd, "--early-stop-min-delta") == "0.01"
    assert _flag_value(cmd, "--early-stop-min-steps") == "10"
    assert build_pipeline_args({"early_stopping": False, "loss_patience": 50}) == ["--no-early-stopping"]


def test_validation_rejects_bad_values_but_allows_unset():
    validate_pipeline_config({"fps": None, "max_steps": None})
    for bad in ({"fps": 0}, {"fps": -1.0}, {"max_steps": 0}, {"data_factor": "x"}):
        with pytest.raises(ValueError):
            validate_pipeline_config(bad)


# --- wrapper.TrainingManager -----------------------------------------------------------------

def test_wrapper_ace_zero_minimal(manager):
    cmd = manager.build_ace_zero_command("video.mp4")
    for flag in AUTO_FLAGS:
        assert flag not in cmd, flag
    assert cmd[3] == "video.mp4"


def test_wrapper_ace_zero_explicit(manager):
    manager.training_config.update({"fps": 2, "max_steps": 300, "depth_model": "midas",
                                    "colmap_input": "/data/x"})
    cmd = manager.build_ace_zero_command("video.mp4")
    assert _flag_value(cmd, "--fps") == "2"
    assert _flag_value(cmd, "--max_steps") == "300"
    assert _flag_value(cmd, "--depth-model") == "midas"
    assert _flag_value(cmd, "--colmap-input") == "/data/x"


def test_wrapper_simple_minimal_and_explicit(manager):
    # simple_trainer has no auto strategy: unset values use the wrapper fallback,
    # not the trainer's own 30k-step / data_factor-4 defaults.
    cmd = manager.build_simple_command("/data")
    assert _flag_value(cmd, "--max_steps") == "7000"
    assert _flag_value(cmd, "--data_factor") == "2"
    assert _flag_value(cmd, "--eval_steps") == "7000"
    assert _flag_value(cmd, "--save_steps") == "7000"
    manager.training_config["max_steps"] = 300
    cmd = manager.build_simple_command("/data")
    assert _flag_value(cmd, "--max_steps") == "300"
    assert _flag_value(cmd, "--eval_steps") == "300"
    assert _flag_value(cmd, "--save_steps") == "300"


def test_wrapper_overrides_do_not_leak_between_runs(manager, monkeypatch):
    monkeypatch.setattr(manager, "_run_ace_zero_pipeline", lambda video_path: True)
    monkeypatch.setattr(manager, "get_video_fps", lambda video_path: 30.0)
    manager.log_file_path = manager.output_dir / "log.txt"
    manager.output_dir.mkdir(parents=True)
    manager.start_training("v.mp4", config_override={"fps": 60, "max_steps": 300})
    assert manager.training_config["fps"] == 30.0  # capped to the video's rate
    manager.status = "idle"
    manager.start_training("v.mp4")
    assert "fps" not in manager.training_config
    assert "max_steps" not in manager.training_config


def test_wrapper_rejects_invalid_override(manager):
    with pytest.raises(ValueError):
        manager.start_training("v.mp4", config_override={"fps": 0})
    assert manager.status == "idle"


# --- log parsing / regex early stopping ------------------------------------------------------

@pytest.mark.parametrize("line,step", [
    ("loss=0.123| sh degree=3| :  12%|#2        | 850/7000 [00:30<03:40, 27.9it/s]", 850),
    ("loss=0.045| sh degree=3| : 100%|██████████| 7000/7000 [04:10<00:00, 27.9it/s]", 7000),
    ("Step 1234: loss=0.05", 1234),
    ("Train Step 42/300 loss: 0.2", 42),
    ("Loading images...", None),
])
def test_parse_training_step(line, step):
    assert parse_training_step(line) == step


def test_parse_training_loss():
    assert parse_training_loss("loss=0.123| sh degree=3|") == pytest.approx(0.123)
    assert parse_training_loss("Loss: 1e-3") == pytest.approx(1e-3)
    assert parse_training_loss("no metric here") is None


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def test_regex_early_stop_counts_trainer_steps_not_lines(manager):
    manager.regex_early_stop = True
    manager.training_config.update({"loss_patience": 100, "min_steps": 0})
    manager.process = _FakeProc()
    (manager.output_dir / "results" / "ckpts").mkdir(parents=True)
    (manager.output_dir / "results" / "ckpts" / "ckpt_99.pt").write_text("x")
    # Many tqdm refreshes of the same few steps must not trip patience.
    for _ in range(500):
        manager._check_early_stopping("loss=0.500| : 1%| | 10/7000 [00:01<1:00, 9it/s]")
    assert not manager.process.terminated
    manager._check_early_stopping("loss=0.500| : 2%| | 111/7000 [00:02<1:00, 9it/s]")
    assert manager.process.terminated and manager.early_stopped


def test_regex_early_stop_waits_for_saved_output(manager):
    manager.regex_early_stop = True
    manager.training_config.update({"loss_patience": 100, "min_steps": 0})
    manager.process = _FakeProc()
    for step in range(0, 5000, 100):
        manager._check_early_stopping(f"loss=0.500| : | {step}/7000 [00:01<1:00, 9it/s]")
    assert not manager.process.terminated


def test_prune_flag_both_ways():
    assert build_pipeline_args({"prune": True}) == ["--prune"]
    assert build_pipeline_args({"prune": False}) == ["--no-prune"]


def test_regex_early_stop_inactive_by_default(manager):
    manager.process = _FakeProc()
    for step in range(0, 10000, 100):
        manager._check_early_stopping(f"loss=0.500| : | {step}/10000 [00:01<1:00, 9it/s]")
    assert not manager.process.terminated


# --- server ----------------------------------------------------------------------------------

def test_server_minimal_request_omits_auto_flags(server):
    config = server.TrainRequest().dict()
    cmd = server.build_pipeline_command("video.mp4", config, output_dir=Path("/tmp/out"))
    for flag in AUTO_FLAGS:
        assert flag not in cmd, flag


def test_server_explicit_request_passes_values(server):
    req = server.TrainRequest(fps=1.5, max_steps=300, data_factor=2, opacity_reg=0.0,
                              scale_reg=0.0, sh_degree=3, depth_model="depth_anything",
                              pose_opt=True, early_stopping=False)
    cmd = server.build_pipeline_command("video.mp4", req.dict(), output_dir=Path("/tmp/out"))
    assert _flag_value(cmd, "--fps") == "1.5"
    assert _flag_value(cmd, "--max_steps") == "300"
    assert _flag_value(cmd, "--data_factor") == "2"
    assert _flag_value(cmd, "--opacity_reg") == "0.0"
    assert _flag_value(cmd, "--scale_reg") == "0.0"
    assert _flag_value(cmd, "--sh_degree") == "3"
    assert _flag_value(cmd, "--depth-model") == "depth_anything"
    assert "--pose-opt" in cmd and "--no-early-stopping" in cmd


def test_server_uses_shared_builder(server):
    assert server.build_pipeline_args is wrapper.build_pipeline_args
