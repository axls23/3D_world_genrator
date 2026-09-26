"""Auto-discovered parameter strategies.

Every module in this package is imported automatically. A module contributes strategies by
defining either module-level NAME / STAGE / derive, or a PARAMS list of objects (e.g.
`Param`) with those three attributes:

    NAME:   the PipelineConfig attribute it sets, e.g. "CAP_MAX"
    STAGE:  one of STAGES - when the pipeline asks for it (signals available by then:
            pre_ace   -> video.*, gpu.*
            post_ace  -> + ace.*, frames.*, points.count
            pre_train -> + gpu.* refreshed right before training
            loop      -> + train.val, difix.rounds (once per feedback-loop iteration))
    derive: derive(profile, config) -> Decision | None   (None = keep the fallback)

A strategy never overrides a value the user set explicitly (config.USER_SET).
Stdlib-only and py3.8-compatible.
"""

import importlib
import logging
import pkgutil
from collections import namedtuple

from ..resolve import Decision, record

logger = logging.getLogger("hypersplat.params")

STAGES = ("pre_ace", "post_ace", "pre_train", "loop")
Param = namedtuple("Param", ["NAME", "STAGE", "derive"])


def discover():
    """All strategies from modules in this package, as Param tuples."""
    params = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as e:
            logger.warning(f"[param] could not load strategy module {info.name}: {e}")
            continue
        if hasattr(module, "PARAMS"):
            params.extend(Param(p.NAME, p.STAGE, p.derive) for p in module.PARAMS)
        elif all(hasattr(module, a) for a in ("NAME", "STAGE", "derive")):
            params.append(Param(module.NAME, module.STAGE, module.derive))
    for p in params:
        if p.STAGE not in STAGES:
            raise ValueError(f"Strategy {p.NAME} has unknown stage {p.STAGE!r}")
    return params


def apply_stage(stage, profile, config):
    """Run every strategy for `stage`, set the config attributes, record decisions."""
    user_set = getattr(config, "USER_SET", set())
    for p in discover():
        if p.STAGE != stage or p.NAME in user_set:
            continue
        try:
            decision = p.derive(profile, config)
        except Exception as e:
            logger.warning(f"[param] {p.NAME}: strategy failed ({e}); keeping "
                           f"{getattr(config, p.NAME, None)!r}")
            continue
        if decision is None:
            continue
        setattr(config, p.NAME, decision.value)
        record(p.NAME, decision, profile)
    if profile is not None:
        profile.save()
