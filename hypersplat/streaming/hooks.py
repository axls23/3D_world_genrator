"""Live runtime hooks: the running stream session and trainer become Kafka producers without a
line of their code changing (Policy & Process Map rows 2, 3, 4, 7; constraints "training speed
comes first" and "keep the old path as a fallback").

    ACE sidecar / CUDA ring --StreamSession._on_event--> [hook] --CameraPose------> TOPIC_POSES
    trainer loop --SummaryWriter.add_scalar("train/*")--> [hook] --TrainingSnapshot--> TOPIC_TRAINING

install() patches both in place, now if the module is already imported, else from a
sys.meta_path post-import hook. Each hook calls the original first and only then does a few
microseconds of conversion plus a non-blocking producer.Sender.send(); nothing it does can raise
into the ACE reader, the ring thread or the training loop. The poses_final.txt hand-off is not
touched (sources.py tails that file for the batch path).

Activated by autohook/sitecustomize.py when HYPERSPLAT_STREAMING=1 (see live.sh).
py3.8-compatible: the ACE env starts Python with the same PYTHONPATH.
"""

import functools
import logging
import math
import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

from . import contract as c

logger = logging.getLogger(__name__)

SESSION_MODULE = "hypersplat.stream.session"
TB_MODULE = "torch.utils.tensorboard.writer"
POSE_KINDS = ("pose", "reloc", "map")

# live.sh -> pipeline process: run_stream_demo.sh rejects --trainer-arg, so the trainer options
# the hooks need (tb_every) ride in this variable and are appended to the pipeline's argv.
TRAINER_ARGS_ENV = "HYPERSPLAT_STREAMING_TRAINER_ARGS"
PIPELINE_SCRIPT = "automated_intelligent_pipeline.py"

_WRAPPED = "_hypersplat_streaming_hook"
_TAP = "_hypersplat_tap"
_START = time.strftime("%Y%m%d-%H%M%S")
_lock = threading.Lock()
_pub_lock = threading.Lock()    # seq hand-out + enqueue as one step (see publish_rows)
_job = None                     # type: Optional[str]
_focal = None                   # type: Optional[float]  # last positive focal on a map row
_unregistered = {}              # type: Dict[str, int]
_warned = set()
_sender_factory = None          # type: Optional[Callable[[], object]]


def _sender():
    if _sender_factory is not None:
        return _sender_factory()
    from .producer import get_sender  # lazy: kafka-python loads on the first event only

    return get_sender()


def _warn_once(what: str) -> None:
    if what not in _warned:
        _warned.add(what)
        logger.warning("[streaming] %s hook failed; the run is unaffected", what, exc_info=True)


# ------------------------------------------------------------------------------------ job id
def _result_dir_from_argv(argv: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for flag in ("--result_dir", "--result-dir"):
            if a == flag and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(flag + "="):
                return a[len(flag) + 1:]
    return None


def _name_of_log_dir(log_dir: str) -> str:
    # the trainer writes to <result_dir>/tb
    path = os.path.normpath(str(log_dir))
    if os.path.basename(path) == "tb":
        path = os.path.dirname(path)
    return os.path.basename(path)


def job_id(log_dir: Optional[str] = None) -> str:
    """HYPERSPLAT_JOB_ID, else ``<result_dir basename>-<process start time>``.

    Fixed on first use so the session's poses and the trainer's snapshots (same process) share
    it: the trainer's --result_dir is in argv before its SummaryWriter logs anything.
    """
    global _job
    if _job is None:
        with _lock:
            if _job is None:
                name = os.environ.get("HYPERSPLAT_JOB_ID")
                if not name:
                    rd = _result_dir_from_argv(sys.argv)
                    if rd:
                        base = _name_of_log_dir(rd)
                    elif log_dir:
                        base = _name_of_log_dir(log_dir)
                    else:
                        base = os.path.basename(os.getcwd()) or "hypersplat"
                    name = "%s-%s" % (base, _START)
                _job = name
    return _job


# ------------------------------------------------------------------------------ pose hook
def unregistered(job: Optional[str] = None) -> int:
    """Frames ACE could not localize (null w2c): a normal outcome, not a drop."""
    with _lock:
        return _unregistered.get(job or job_id(), 0)


def _count_unregistered(job: str) -> None:
    with _lock:
        n = _unregistered[job] = _unregistered.get(job, 0) + 1
    if n == 1 or n % 100 == 0:
        logger.info("[streaming] %d unregistered frame(s) for %s (not sent)", n, job)


def publish_rows(kind: str, rows: List[Dict]) -> None:
    """Send one CameraPose per registered row of a pose / reloc / map event. Never raises."""
    global _focal
    if kind not in POSE_KINDS:
        return  # error / exit / anything else is the session's business
    try:
        sender = _sender()
        job = job_id()
        if kind == "map":
            for row in rows:
                try:
                    f = float(row.get("focal"))  # numpy scalars too
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f) and f > 0:
                    _focal = f
        for row in rows:
            # checked here, not only via Unregistered, so an unregistered frame takes no seq
            # (a seq gap means a lost event to the processor)
            if row.get("w2c") is None:
                _count_unregistered(job)
                continue
            # _on_event runs on the sidecar reader thread and on the ring thread. The seq is handed out and
            # the event queued under one lock: otherwise seq 6 can be queued before seq 5, and the
            # processor skips 5 as late (no re-ordering buffer), losing a pose. send() never blocks, so
            # the lock is held for microseconds.
            with _pub_lock:
                try:
                    ev = c.pose_from_event_row(row, job, sender.next_seq(job), focal=_focal,
                                               dropped=sender.dropped(job))
                except c.Unregistered:
                    _count_unregistered(job)
                    continue
                except c.InvalidEvent as e:
                    sender.drop(job, str(e))
                    continue
                sender.send(ev)
    except Exception:
        _warn_once("pose")


HEALTH_EVERY_S = 1.0
_HEALTH_FIELDS = ("frames_read", "frames_dropped", "keyframes_offered", "keyframes_accepted", "keyframes",
                  "registered", "unregistered", "evicted", "reconnects", "localize_ms", "depth_rejected",
                  "depth_backlog", "store_mb")


def start_health(session, every: float = HEALTH_EVERY_S) -> threading.Thread:
    """Sample session.stats() into one StreamHealth per ``every`` seconds until the session stops.

    This is the head of the pipeline (source queue, gate, store, localisation) that the pose and training
    taps cannot see: it runs from session start, before any pose or training step exists. A failing stats
    call skips that sample; nothing here can raise into the session.
    """
    stop = getattr(session, "_stop", None) or threading.Event()

    def run():
        while not stop.is_set():
            try:
                stats = session.stats()
                sender = _sender()
                job = job_id()
                data = {k: stats.get(k) for k in _HEALTH_FIELDS if stats.get(k) is not None}
                try:
                    phase = session.phase()[0]
                except Exception:
                    phase = ""
                with _pub_lock:
                    sender.send(c.validate(c.StreamHealth, dict(data, job_id=job, seq=sender.next_seq(job + "#health"),
                                                                ts=time.time(), phase=phase)))
            except c.InvalidEvent:
                pass
            except Exception:
                _warn_once("health")
            stop.wait(every)

    t = threading.Thread(target=run, name="hypersplat-stream-health", daemon=True)
    t.start()
    return t


def _patch_session(mod) -> None:
    cls = mod.StreamSession
    orig = cls._on_event
    if getattr(orig, _WRAPPED, False):
        return

    orig_start = cls.start

    @functools.wraps(orig_start)
    def start(self, *args, **kwargs):
        out = orig_start(self, *args, **kwargs)
        try:
            start_health(self)
        except Exception:
            _warn_once("health")
        return out

    setattr(start, _WRAPPED, True)
    cls.start = start

    @functools.wraps(orig)
    def _on_event(self, kind, rows):
        try:
            return orig(self, kind, rows)
        finally:  # the pose happened even if the session's handler failed on it
            publish_rows(kind, rows)

    setattr(_on_event, _WRAPPED, True)
    cls._on_event = _on_event


# -------------------------------------------------------------------------- training hook
class _TrainTap(object):
    """Collects one step's train/* scalars and emits them as one TrainingSnapshot."""

    def __init__(self, log_dir: Optional[str]):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.step = None            # type: Optional[int]
        self.t = 0.0                # monotonic time the pending step's first scalar arrived
        self.scalars = {}           # type: Dict[str, float]
        self.done_step = None       # type: Optional[int]
        self.rate_from = None       # (step, t) of the last emitted snapshot
        self.sticky = {}            # type: Dict[str, float]  latest PSNR / Difix scalars, carried forward
        self.sticky_at_step = {}    # type: Dict[str, float]  what was known when the pending step began

    def add(self, tag: str, value, step) -> None:
        if step is None:
            return
        if tag in c.STICKY_TAGS:
            # logged at its own step (an evaluation, a probe, a Difix round): remember it for the next
            # snapshot rather than letting it start a snapshot of its own
            with self.lock:
                self.sticky[tag] = float(value)
            return
        if tag not in c.TB_TAGS:
            return
        step = int(step)
        with self.lock:
            if step == self.done_step:
                return  # a late tag for a step already sent on flush
            if self.step is not None and step != self.step:
                self._emit()
            if self.step is None:
                self.step, self.t = step, time.monotonic()
                self.sticky_at_step = dict(self.sticky)
            self.scalars[tag] = float(value)

    def flush(self) -> None:
        with self.lock:
            self._emit()

    def _emit(self) -> None:
        if self.step is None:
            return
        step, t, scalars = self.step, self.t, dict(self.scalars, **self.sticky_at_step)
        self.step, self.scalars, self.done_step = None, {}, step
        rate = None
        if self.rate_from is not None:
            ds, dt = step - self.rate_from[0], t - self.rate_from[1]
            if ds > 0 and dt > 0:
                rate = ds / dt
        self.rate_from = (step, t)
        sender = _sender()
        job = job_id(self.log_dir)
        try:
            ev = c.snapshot_from_scalars(step, scalars, job, sender.next_seq(job),
                                         steps_per_s=rate, dropped=sender.dropped(job))
        except c.InvalidEvent as e:  # e.g. a step that logged train/mem but not train/loss
            sender.drop(job, str(e))
            return
        sender.send(ev)


def _tap(writer) -> _TrainTap:
    tap = writer.__dict__.get(_TAP)
    if tap is None:
        tap = writer.__dict__.setdefault(_TAP, _TrainTap(getattr(writer, "log_dir", None)))
    return tap


def _patch_tb(mod) -> None:
    cls = mod.SummaryWriter
    if getattr(cls.add_scalar, _WRAPPED, False):
        return
    orig_add = cls.add_scalar

    @functools.wraps(orig_add)
    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        out = orig_add(self, tag, scalar_value, global_step, *args, **kwargs)
        try:
            if tag in c.TB_TAGS or tag in c.STICKY_TAGS:
                _tap(self).add(tag, scalar_value, global_step)
        except Exception:
            _warn_once("training")
        return out

    def _then_emit(orig):
        @functools.wraps(orig)
        def end(self, *args, **kwargs):
            try:
                return orig(self, *args, **kwargs)
            finally:
                try:
                    tap = self.__dict__.get(_TAP)
                    if tap is not None:
                        tap.flush()
                except Exception:
                    _warn_once("training")
        return end

    # the trainer calls flush() right after each step's train/* scalars: emit there, no wait
    # for the next step
    for name, fn in (("add_scalar", add_scalar), ("flush", _then_emit(cls.flush)),
                     ("close", _then_emit(cls.close))):
        setattr(fn, _WRAPPED, True)
        setattr(cls, name, fn)


# ------------------------------------------------------------------------- post-import hook
_PATCHERS = {SESSION_MODULE: _patch_session, TB_MODULE: _patch_tb}
_pending = set()  # module names still waiting to be imported


def _apply(name: str, module) -> None:
    _pending.discard(name)
    try:
        _PATCHERS[name](module)
        logger.info("[streaming] hooked %s", name)
    except Exception:
        _warn_once(name)


class _PostImportFinder(object):
    """sys.meta_path entry: lets the real finders locate a target module, then patches it right
    after its body has run. Costs one set lookup per import otherwise."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _pending:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        orig_exec = loader.exec_module
        had_own = "exec_module" in getattr(loader, "__dict__", {})

        def exec_module(module):
            # one-shot: put the loader back first, in case it is shared (zipimport)
            if had_own:
                loader.exec_module = orig_exec
            else:
                del loader.exec_module
            orig_exec(module)
            if module.__name__ == fullname:
                _apply(fullname, module)

        # Patch this spec's loader instance rather than swap in a wrapper loader, so
        # get_source / get_data (inspect, linecache, torch.jit) keep seeing the real one.
        try:
            loader.exec_module = exec_module
        except Exception:  # loader without __dict__: wrap it instead
            spec.loader = _ExecThenPatch(loader, fullname)
        return spec

    def invalidate_caches(self):
        pass


class _ExecThenPatch(object):
    def __init__(self, loader, fullname):
        self._loader, self._fullname = loader, fullname

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _apply(self._fullname, module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


_finder = _PostImportFinder()


def install(sender_factory: Optional[Callable[[], object]] = None) -> None:
    """Patch the session and the trainer's SummaryWriter, now or when they are imported.

    Idempotent. ``sender_factory`` replaces producer.get_sender (tests pass a FakeSender).
    """
    global _sender_factory
    if sender_factory is not None:
        _sender_factory = sender_factory
    else:
        from . import producer  # noqa: F401  stdlib only; keeps the import off the first event
    for name in _PATCHERS:
        mod = sys.modules.get(name)
        if mod is not None:
            _apply(name, mod)
        else:
            _pending.add(name)
    if _pending and _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)


def uninstall() -> None:
    """Restore the original methods and forget all hook state (tests)."""
    global _sender_factory, _job, _focal
    session = sys.modules.get(SESSION_MODULE)
    targets = [(getattr(session, "StreamSession", None), ("_on_event", "start"))]
    tb = sys.modules.get(TB_MODULE)
    targets.append((getattr(tb, "SummaryWriter", None), ("add_scalar", "flush", "close")))
    for cls, names in targets:
        for name in names:
            fn = getattr(cls, name, None) if cls is not None else None
            if getattr(fn, _WRAPPED, False):
                setattr(cls, name, fn.__wrapped__)
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    _pending.clear()
    _unregistered.clear()
    _warned.clear()
    _sender_factory, _job, _focal = None, None, None


def extend_pipeline_argv(argv: Optional[List[str]] = None) -> None:
    """In the pipeline process started by run_stream_demo.sh, append ``--trainer-arg KV`` for each
    KEY=VALUE in $HYPERSPLAT_STREAMING_TRAINER_ARGS (manager.py passes them to the trainer)."""
    argv = sys.argv if argv is None else argv
    extra = os.environ.get(TRAINER_ARGS_ENV, "").split()
    if not extra or not argv or os.path.basename(argv[0]) != PIPELINE_SCRIPT:
        return
    given = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--trainer-arg"]
    given += [a[len("--trainer-arg="):] for a in argv if a.startswith("--trainer-arg=")]
    keys = {kv.split("=", 1)[0].lstrip("-") for kv in given}
    for kv in extra:
        if kv.split("=", 1)[0] not in keys:  # an explicit --trainer-arg wins
            argv.extend(["--trainer-arg", kv])


def stats() -> Dict[str, object]:
    """Counters for the current job (tests, end-of-run checks)."""
    job, s = job_id(), _sender()
    return {"job_id": job, "unregistered": unregistered(job), "sent": getattr(s, "sent", None),
            "dropped": s.dropped(job)}
