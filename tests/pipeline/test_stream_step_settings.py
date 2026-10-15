"""Stream step schedule (manager.stream_step_settings): CPU-only, no trainer launch.

A live run without --steps grows past any fixed budget, so it must eval/save/export only on
its final step; explicit --steps and file sources keep the fixed [steps] schedule.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from hypersplat.pipeline import manager
from hypersplat.pipeline.manager import (FINAL_STEP_ONLY, LIVE_INITIAL_STEPS,
                                         stream_step_settings)

SENTINEL = str(FINAL_STEP_ONLY)


def _flag(args, name):
    """Value after a trainer flag, or None if the flag is absent."""
    return args[args.index(name) + 1] if name in args else None


def test_dynamic_live_evals_saves_exports_only_on_final_step(make_config):
    s = stream_step_settings(make_config(), "url")
    assert s.max_steps == LIVE_INITIAL_STEPS
    assert s.eval_steps == [FINAL_STEP_ONLY]
    assert s.save_steps == [FINAL_STEP_ONLY]
    assert _flag(s.extra_args, "--ply_steps") == SENTINEL
    # The trainer's [7000, 30000] defaults would fire mid-stream once the run grows past 7k
    assert FINAL_STEP_ONLY > 30_000
    assert "--stream_max_steps_cap" not in s.extra_args


def test_dynamic_live_warmup_matches_old_10k_schedule(make_config):
    # Not 300 (15% of the 2k starting budget): the run grows well past it
    assert stream_step_settings(make_config(), "camera").warmup == 1500


@pytest.mark.parametrize("kind", ["url", "camera", "flight", "android"])
def test_explicit_steps_live_is_unchanged(make_config, kind):
    cfg = make_config(TRAINING_MAX_STEPS=5000, USER_SET={"TRAINING_MAX_STEPS"})
    s = stream_step_settings(cfg, kind)
    assert s.max_steps == 5000
    assert s.eval_steps == [5000] and s.save_steps == [5000]
    assert s.warmup == 750
    assert s.extra_args == ["--ply_steps", "5000", "--stream_max_steps_cap", "5000"]


def test_file_source_is_unchanged(make_config):
    s = stream_step_settings(make_config(TRAINING_MAX_STEPS=7000), "file")
    assert s.max_steps == 7000
    assert s.eval_steps == [7000] and s.save_steps == [7000]
    assert s.warmup == 1050
    assert s.extra_args == ["--ply_steps", "7000", "--stream_max_steps_cap", "7000"]


def test_user_set_values_win_in_dynamic_mode(make_config):
    cfg = make_config(TRAINING_EVAL_STEPS=[3000], TRAINING_SAVE_STEPS=[4000],
                      POSE_OPT_WARMUP=200,
                      USER_SET={"TRAINING_EVAL_STEPS", "TRAINING_SAVE_STEPS", "POSE_OPT_WARMUP"})
    s = stream_step_settings(cfg, "url")
    assert s.max_steps == LIVE_INITIAL_STEPS
    assert s.eval_steps == [3000] and s.save_steps == [4000] and s.warmup == 200
    assert _flag(s.extra_args, "--ply_steps") == SENTINEL


def test_user_set_values_win_with_explicit_steps(make_config):
    cfg = make_config(TRAINING_MAX_STEPS=5000, TRAINING_EVAL_STEPS=[1000, 5000],
                      POSE_OPT_WARMUP=0,
                      USER_SET={"TRAINING_MAX_STEPS", "TRAINING_EVAL_STEPS", "POSE_OPT_WARMUP"})
    s = stream_step_settings(cfg, "url")
    assert s.eval_steps == [1000, 5000] and s.save_steps == [5000] and s.warmup == 0


def test_settings_do_not_mutate_config(make_config):
    cfg = make_config()
    before = dict(vars(cfg))
    stream_step_settings(cfg, "url")
    assert vars(cfg) == before


class _Launched(Exception):
    pass


def _trainer_cmd(tmp_path, monkeypatch, argv):
    """Trainer command run_stream builds from real CLI args, with the launch mocked out."""
    captured = {}

    def fake_run(self, video_path):
        captured["config"] = self.config

    with mock.patch.object(sys, "argv", ["manager", *argv, "--output_dir", str(tmp_path)]), \
            mock.patch.object(manager.IntelligentPipeline, "run", fake_run):
        manager.main()
    config = captured["config"]
    config.CAP_MAX = 100_000
    config.USER_SET.add("CAP_MAX")

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        raise _Launched

    from hypersplat.pipeline.params import PROFILE_ENV
    monkeypatch.setenv(PROFILE_ENV, "")
    pipe = manager.IntelligentPipeline(config)
    with mock.patch.object(manager.IntelligentPipeline, "_record_gpu_signals"), \
            mock.patch.object(manager.subprocess, "Popen", fake_popen), \
            pytest.raises(_Launched):
        pipe.run_stream(argv[0])
    return captured["cmd"], config


def _flag_values(cmd, name):
    i = cmd.index(name) + 1
    out = []
    while i < len(cmd) and not cmd[i].startswith("--"):
        out.append(cmd[i])
        i += 1
    return out


def test_run_stream_uses_helper_for_dynamic_live(tmp_path, monkeypatch):
    cmd, config = _trainer_cmd(tmp_path, monkeypatch, ["rtsp://cam.local/live", "--stream"])
    assert _flag(cmd, "--max_steps") == str(LIVE_INITIAL_STEPS)
    for name in ("--eval_steps", "--save_steps", "--ply_steps"):
        assert _flag_values(cmd, name) == [SENTINEL], name
    assert "--stream_max_steps_cap" not in cmd
    assert config.POSE_OPT_WARMUP == 1500


def test_run_stream_explicit_steps_unchanged(tmp_path, monkeypatch):
    cmd, config = _trainer_cmd(tmp_path, monkeypatch,
                               ["rtsp://cam.local/live", "--stream", "--max_steps", "5000"])
    assert _flag(cmd, "--max_steps") == "5000"
    for name in ("--eval_steps", "--save_steps", "--ply_steps", "--stream_max_steps_cap"):
        assert _flag_values(cmd, name) == ["5000"], name
    assert config.POSE_OPT_WARMUP == 750
