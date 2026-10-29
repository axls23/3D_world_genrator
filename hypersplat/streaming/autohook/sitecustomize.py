"""Autoload of the live streaming hooks (Policy & Process Map rows 2, 3, 7).

Python imports ``sitecustomize`` at startup from the first sys.path entry that has one, so
putting this directory on PYTHONPATH (live.sh does) reaches every Python process the stream demo
starts: the pipeline, the trainer subprocess it spawns, and the ACE sidecar (py3.8).

With HYPERSPLAT_STREAMING=1 it calls hooks.install(); otherwise it only chains. It then runs
the sitecustomize this one shadows, and nothing in here may fail interpreter startup.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _same(a, b):
    try:
        return os.path.realpath(a or ".") == os.path.realpath(b)
    except Exception:
        return False


def _activate():
    # cheap exit before any import when streaming is off (every Python process pays this)
    if os.environ.get("HYPERSPLAT_STREAMING") != "1":
        return
    root = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    if not any(_same(p, root) for p in sys.path):
        sys.path.append(root)  # appended: never shadows what the process would import anyway
    from hypersplat.streaming import contract

    if contract.STREAMING_ENABLED:
        from hypersplat.streaming import hooks

        hooks.install()
        hooks.extend_pipeline_argv()


def _chain():
    """Run the next sitecustomize on sys.path (venv / distro ones) that this file shadows."""
    from importlib.machinery import PathFinder
    from importlib.util import module_from_spec

    rest = [p for p in sys.path if not _same(p, _HERE)]
    spec = PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None or _same(os.path.dirname(spec.origin or ""), _HERE):
        return
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.setdefault("_hypersplat_chained_sitecustomize", mod)


for _step in (_activate, _chain):
    try:
        _step()
    except Exception as _e:  # never break interpreter startup
        try:
            sys.stderr.write("[streaming] sitecustomize %s skipped: %r\n" % (_step.__name__, _e))
        except Exception:
            pass
