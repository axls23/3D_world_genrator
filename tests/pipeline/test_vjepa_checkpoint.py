"""CPU tests for the V-JEPA checkpoint startup check (examples/vjepa_checkpoint.py)."""

import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from vjepa_checkpoint import (  # noqa: E402
    VJEPA_CHECKPOINT_DEFAULT,
    VJEPA_CHECKPOINT_ENV,
    VJEPA_CHECKPOINT_URL,
    default_vjepa_checkpoint,
    missing_vjepa_checkpoint,
)


def test_default_path_without_env(monkeypatch):
    monkeypatch.delenv(VJEPA_CHECKPOINT_ENV, raising=False)
    assert default_vjepa_checkpoint() == VJEPA_CHECKPOINT_DEFAULT


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv(VJEPA_CHECKPOINT_ENV, "/elsewhere/vitl.pt")
    assert default_vjepa_checkpoint() == "/elsewhere/vitl.pt"


def test_empty_env_keeps_default(monkeypatch):
    monkeypatch.setenv(VJEPA_CHECKPOINT_ENV, "")
    assert default_vjepa_checkpoint() == VJEPA_CHECKPOINT_DEFAULT


def test_missing_file_message(tmp_path):
    path = str(tmp_path / "nope.pt")
    msg = missing_vjepa_checkpoint(True, "vjepa", path)
    assert msg is not None
    assert path in msg
    assert VJEPA_CHECKPOINT_URL in msg
    assert VJEPA_CHECKPOINT_ENV in msg
    assert "latent_model=feature" in msg


@pytest.mark.parametrize("checkpoint", [None, ""])
def test_unset_path_is_missing(checkpoint):
    assert "unset" in missing_vjepa_checkpoint(True, "vjepa", checkpoint)


def test_directory_is_not_a_checkpoint(tmp_path):
    assert missing_vjepa_checkpoint(True, "vjepa", str(tmp_path)) is not None


def test_truncated_file_is_rejected(tmp_path):
    ckpt = tmp_path / "vitl.pt"
    ckpt.write_bytes(b"PK\x03\x04 partial download")
    assert "truncated" in missing_vjepa_checkpoint(True, "vjepa", str(ckpt))


def test_torch_checkpoint_passes(tmp_path):
    ckpt = tmp_path / "vitl.pt"
    torch.save({"target_encoder": {}, "predictor": {}}, ckpt)
    assert missing_vjepa_checkpoint(True, "vjepa", str(ckpt)) is None


@pytest.mark.parametrize("latent_loss, latent_model", [(False, "vjepa"), (True, "feature"), (False, "feature")])
def test_not_needed_without_vjepa_latent_loss(latent_loss, latent_model):
    assert missing_vjepa_checkpoint(latent_loss, latent_model, "/nonexistent/vitl.pt") is None
