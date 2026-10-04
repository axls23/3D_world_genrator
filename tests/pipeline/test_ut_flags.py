"""UT (3DGUT) flag validation in the pipeline manager: streams keep UT, batch runs keep
pose optimization (gsplat's UT path has no pose gradients)."""

import sys
from unittest import mock

import pytest

from hypersplat.pipeline import manager


def _config(tmp_path, *flags):
    captured = {}

    class _Pipeline:
        def __init__(self, config):
            captured["config"] = config

        def run(self, _):
            pass

    argv = ["automated_intelligent_pipeline.py", "in.mp4", "--output_dir", str(tmp_path), *flags]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(manager, "IntelligentPipeline", _Pipeline):
        manager.main()
    return captured["config"]


def test_stream_ut_keeps_ut_and_drops_pose_opt(tmp_path):
    c = _config(tmp_path, "--stream", "--with_ut", "--stream-pose-opt")
    assert c.WITH_UT and c.WITH_EVAL3D  # UT implies eval3d
    assert not c.POSE_OPT and not c.STREAM_POSE_OPT


def test_batch_ut_yields_to_pose_opt(tmp_path):
    c = _config(tmp_path, "--with_ut")
    assert c.POSE_OPT and not c.WITH_UT and not c.WITH_EVAL3D


def test_batch_ut_without_pose_opt(tmp_path):
    c = _config(tmp_path, "--with_ut", "--no-pose-opt")
    assert c.WITH_UT and c.WITH_EVAL3D and not c.POSE_OPT


def test_stream_without_ut_untouched(tmp_path):
    c = _config(tmp_path, "--stream", "--stream-pose-opt")
    assert not c.WITH_UT and c.STREAM_POSE_OPT
