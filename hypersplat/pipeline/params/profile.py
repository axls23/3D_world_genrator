"""SceneProfile: the measured signals a run has gathered so far, shared by all stages.

Each pipeline stage records what it measured (video stats, ACE-Zero outputs, GPU memory,
training metrics, ...) and parameter strategies read from it instead of hard-coded guesses.
Stored as JSON at <OUTPUT_BASE>/scene_profile.json so subprocesses (e.g. the ACE-Zero point
generator, which runs in a separate py3.8 env) can read it via $HYPERSPLAT_PROFILE.

Stdlib-only and py3.8-compatible.

Key schema (dotted paths; strategies should only rely on these):
    video.fps, video.frame_count, video.width, video.height, video.duration, video.avg_motion
    frames.count, frames.width, frames.height          extracted frames used for training
    ace.focal_median                                   ACE-Zero refined focal (original px)
    ace.conf                                           {image_name: registration confidence}
    ace.n_registered
    points.count                                       initial point cloud size
    gpu.total_mb, gpu.free_mb
    train.val                                          [{step, psnr, ssim, lpips, num_GS}]
    difix.rounds                                       [{angle, change_psnr, ...}]
    decisions.<CONFIG_NAME>                            {value, strategy, reason}
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

PROFILE_ENV = "HYPERSPLAT_PROFILE"
PROFILE_FILENAME = "scene_profile.json"


class SceneProfile:
    def __init__(self, path: Optional[os.PathLike] = None, data: Optional[dict] = None):
        self.path = Path(path) if path is not None else None
        self.data = data if data is not None else {}

    @classmethod
    def load(cls, path: os.PathLike) -> "SceneProfile":
        """Load from path, or start empty if it does not exist (or is unreadable)."""
        path = Path(path)
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (ValueError, OSError):
                data = {}
        return cls(path, data)

    @classmethod
    def from_env(cls) -> Optional["SceneProfile"]:
        """Profile of the running pipeline, for subprocesses; None when not set."""
        path = os.environ.get(PROFILE_ENV)
        return cls.load(path) if path else None

    def get(self, key: str, default: Any = None) -> Any:
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any, source: Optional[str] = None) -> None:
        parts = key.split(".")
        node = self.data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        if source:
            self.data.setdefault("_sources", {})[key] = source

    def append(self, key: str, item: Any, source: Optional[str] = None) -> None:
        items = self.get(key)
        items = list(items) if isinstance(items, list) else []
        items.append(item)
        self.set(key, items, source)

    def discard(self, *keys: str) -> None:
        """Drop top-level sections (e.g. per-run results) and write without merging."""
        for key in keys:
            self.data.pop(key, None)
        self.save(merge=False)

    def save(self, merge: bool = True) -> None:
        """Write to disk, merging with what is already there (unless merge=False).

        Stage subprocesses (e.g. the ACE-Zero point generator) and in-process helpers may
        hold their own SceneProfile of the same file; a plain overwrite from a stale copy
        would drop their keys. Keys in memory win on conflict, keys only on disk are kept,
        and this object is refreshed with the merged result.
        """
        if self.path is None:
            return
        if merge:
            self.data = _merge(SceneProfile.load(self.path).data, self.data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        os.replace(str(tmp), str(self.path))


def _merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins on conflicting non-dict values."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out
