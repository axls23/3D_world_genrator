"""CPU-only fixtures for dynamic-parameter strategy tests.

The profiles in fixtures/ are real measurements from the 2026-09-26 runs (drone flyover and a
handheld WhatsApp night clip) on an RTX 4050 Laptop (6 GB). `_observed` holds what actually
happened with the old hard-coded values, so tests can assert a strategy avoids that failure.
"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypersplat.pipeline.params import SceneProfile  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def drone_profile():
    """Drone flyover: 1920x1080 source, 40 frames at 640x360, stable ACE-Zero poses."""
    return SceneProfile(data=copy.deepcopy(_load("drone_profile.json")))


@pytest.fixture
def whatsapp_profile():
    """Handheld 480x864 night clip: 139 frames, 1.15M-point cloud that OOM'd at cap 1M."""
    return SceneProfile(data=copy.deepcopy(_load("whatsapp_profile.json")))


@pytest.fixture
def make_config():
    """PipelineConfig stand-in with today's fallbacks; override attributes via kwargs."""
    def _make(**overrides):
        cfg = SimpleNamespace(
            FPS=10.0, DATA_FACTOR=2, TRAINING_MAX_STEPS=7000,
            TRAINING_EVAL_STEPS=[3000, 7000], TRAINING_SAVE_STEPS=[3000, 7000],
            INIT_SCALE=2.5, CAP_MAX=None, MIN_REGISTRATION_CONFIDENCE=1000,
            POSE_OPT_WARMUP=1000, REFINE_LOOPS=3, DIFIX_NUM_VIEWS=24,
            DIFIX_MAX_ANGLE=20.0, DIFIX_PSEUDO_WEIGHT=0.5, TEST_EVERY=8,
            QUALITY_MODE="balanced", USER_SET=set(),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
    return _make
