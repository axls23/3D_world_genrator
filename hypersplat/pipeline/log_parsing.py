"""
Shared tqdm/marker-based training log-line parsing.

This module consolidates the progress-parsing heuristics that were
previously duplicated (near-identically) in:
  - hypersplat.services.api.server.AppState.parse_progress
  - hypersplat.pipeline.wrapper.TrainingManager._parse_log_line

Both call sites match the same markers (e.g. "STEP 1:", "3DGS TRAINING",
"PIPELINE COMPLETE") and the same tqdm-style "it/s]" progress lines in the
same order, but each then applies its own distinct side effects (updating a
``dict``-based job vs. instance attributes / a log deque). This module owns
only the pure string-matching logic; each caller keeps its own thin wrapper
that applies its own side effects based on the returned result.

IMPORTANT: The marker strings, regex/parsing logic, and match ordering here
are copied byte-for-byte from the two original implementations. Do not
change matching behavior here without updating both call sites' expected
behavior, since this is meant to be a pure refactor (no behavior change).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarkerProgressResult:
    """Neutral result of scanning a single log line for known markers."""

    # Extracted "TRAINING_VIEWER_URL:" value, if the line contained one.
    viewer_url: Optional[str] = None

    # Final progress percentage implied by this line, if any marker matched.
    # None means "no progress-affecting marker matched this line".
    progress: Optional[int] = None

    # Whether this line matched the "ACE-ZERO POSE ESTIMATION" marker.
    saw_ace_zero_pose: bool = False

    # Whether this line matched the "3DGS TRAINING" marker.
    saw_3dgs_training: bool = False

    # Whether this line matched a pipeline-complete marker
    # ("PIPELINE COMPLETE" or "PIPELINE SUMMARY").
    is_complete: bool = False


def parse_marker_progress(line: str) -> MarkerProgressResult:
    """Scan a single training log line for known progress markers.

    Mirrors, in the same order, the marker checks previously duplicated in
    ``AppState.parse_progress`` (server.py) and
    ``TrainingManager._parse_log_line`` (wrapper.py). Returns a neutral
    result object; callers are responsible for applying their own
    side effects (e.g. setting a job/status dict field, appending to a log
    buffer, etc).
    """
    result = MarkerProgressResult()

    # Capture Viewer URL
    if "TRAINING_VIEWER_URL:" in line:
        parts = line.split("TRAINING_VIEWER_URL:")
        if len(parts) > 1:
            result.viewer_url = parts[1].strip()

    # ACE-Zero specific progress
    if "ACE-ZERO POSE ESTIMATION" in line:
        result.progress = 15
        result.saw_ace_zero_pose = True
    if "Extracting frames" in line:
        result.progress = 20
    if "Extracted" in line and "frames" in line:
        result.progress = 30
    if "3DGS TRAINING" in line:
        result.progress = 40
        result.saw_3dgs_training = True

    # Legacy progress indicators
    if "STEP 1:" in line:
        result.progress = 10
    if "STEP 2:" in line:
        result.progress = 40
    if "STEP 3:" in line:
        result.progress = 70

    # Training step progress (from tqdm output), e.g:
    # 375/500 [00:38<00:12, 10.25it/s]
    if "it/s]" in line and "/" in line:
        try:
            parts = line.split("|")
            if len(parts) >= 2:
                step_part = parts[-1].strip()
                if "/" in step_part:
                    current, total = step_part.split("/")[0], step_part.split("/")[1].split()[0]
                    current, total = int(current.strip()), int(total.strip())
                    # Map 40-95% to training progress
                    train_progress = (current / total) * 55
                    result.progress = 40 + int(train_progress)
        except:
            pass

    if "PIPELINE COMPLETE" in line or "PIPELINE SUMMARY" in line:
        result.progress = 100
        result.is_complete = True

    return result


def is_repetitive_match(last_line: str, new_line: str) -> bool:
    """Check if new_line is a progress update of the same type as last_line."""
    # Match tqdm style
    if "it/s]" in last_line and "it/s]" in new_line:
        return True
    # Match iteration style
    if "Iteration" in last_line and "Iteration" in new_line:
        return True
    return False
