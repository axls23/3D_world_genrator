"""Dynamic pipeline parameters: measured signals (SceneProfile) -> per-parameter strategies.

See strategies/__init__.py for how to add a strategy. Stdlib-only and py3.8-compatible.
"""

from .profile import PROFILE_ENV, PROFILE_FILENAME, SceneProfile
from .resolve import Decision, record, resolve

__all__ = ["SceneProfile", "PROFILE_ENV", "PROFILE_FILENAME", "Decision", "record", "resolve"]
