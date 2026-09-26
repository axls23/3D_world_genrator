"""Decision resolution: explicit user value > derived strategy value > today's fallback.

Stdlib-only and py3.8-compatible.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("hypersplat.params")


@dataclass
class Decision:
    value: Any
    strategy: str  # short name of the rule that produced it, e.g. "vram-budget"
    reason: str    # human-readable why, including the signals used


def record(name: str, decision: Decision, profile=None) -> None:
    logger.info(f"[param] {name}={decision.value!r} ({decision.strategy}: {decision.reason})")
    if profile is not None:
        profile.set(f"decisions.{name}",
                    {"value": decision.value, "strategy": decision.strategy, "reason": decision.reason})


def resolve(name: str, user_value: Any, derive: Optional[Callable[[], Optional[Decision]]],
            fallback: Any, profile=None) -> Any:
    """Return the value to use for `name` and log/record where it came from.

    user_value: the explicitly set value, or None when the user left it on auto.
    derive: zero-arg callable returning a Decision, or None when signals are missing.
    fallback: the previous hard-coded default.
    """
    if user_value is not None:
        decision = Decision(user_value, "user", "set explicitly")
    else:
        decision = None
        if derive is not None:
            try:
                decision = derive()
            except Exception as e:  # a broken strategy must never break the pipeline
                logger.warning(f"[param] {name}: strategy failed ({e}); using fallback")
        if decision is None:
            decision = Decision(fallback, "fallback", "no signal available")
    record(name, decision, profile)
    return decision.value
