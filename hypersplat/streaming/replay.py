"""Replay producer: streams a recorded run into Kafka at a steady pace (rows 3, 7, 12, 21).

The data-source layer without a GPU: a recordings/<name>.jsonl from recording.py is sent through
producer.Sender exactly as the live hooks would send it, so the processor, models, database and
dashboard see an ordinary job.

    python -m hypersplat.streaming.replay --recording recordings/record1.jsonl \\
        [--rate 5] [--job-id NEW] [--inject shuffle|blur --at N --length K] [--reset] [--loop]

Every replay is a fresh job: job_id becomes replay-<timestamp> (or --job-id), seq is renumbered
with Sender.next_seq and ts is the send time. Pose sources are rewritten to "replay" (scored by
the processor; see --keep-source), since most recorded rows are reloc rows that are not.

Liveness demo (row 21): press p (stdin on a TTY) or send SIGUSR1 to pause / resume; nothing is
sent while paused and the status line says PAUSED.

Ground truth for the anomaly models (row 12), pose events only, counted in pose order from 0:
    shuffle  poses N..N+K-1 take the camera of the pose K positions ahead (even offsets) or
             behind (odd offsets), so every one of them is a camera jump; injected="shuffle"
    blur     poses N..N+K-1 keep their place but lose 90% of their confidence and get a small
             position / rotation jitter; injected="blur"

--reset deletes --job-id's rows from pose_history, training_history and jobs first (no-op for
tables that do not exist yet).
"""

import argparse
import math
import os
import random
import signal
import sys
import threading
import time
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Sequence

from hypersplat.streaming import contract as c

BLUR_CONF_SCALE = 0.1
BLUR_ROT_DEG = 1.0          # sigma of the rotation jitter
BLUR_POS_FRACTION = 0.25    # sigma of the position jitter, as a fraction of the median step
STATUS_EVERY_S = 2.0        # status line period when stderr is not a terminal
RESET_TABLES = ("pose_history", "training_history", "jobs")  # children before the FK parent


# ---------------------------------------------------------------------------------------------
# Recording and injection
# ---------------------------------------------------------------------------------------------

def load(path: str) -> List:
    """Events of a recording in file order; a line that fails validation is skipped."""
    events, bad = [], 0
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = c.decode(line)
            except c.InvalidEvent:
                bad += 1
                continue
            if isinstance(e, (c.CameraPose, c.TrainingSnapshot)):  # anomalies are processor output
                events.append(e)
            else:
                bad += 1
    if bad:
        print("[replay] skipped %d invalid line(s) in %s" % (bad, path), file=sys.stderr)
    return events


_GEOMETRY = ("qw", "qx", "qy", "qz", "tx", "ty", "tz")


def _rotmat(q: Sequence[float]):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _qmul(a: Sequence[float], b: Sequence[float]):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _median_step(poses: Sequence[c.CameraPose]) -> float:
    steps = []
    for a, b in zip(poses, poses[1:]):
        ca, cb = c.camera_center(a), c.camera_center(b)
        steps.append(math.sqrt(sum((ca[i] - cb[i]) ** 2 for i in range(3))))
    steps = sorted(s for s in steps if s > 0)
    return steps[len(steps) // 2] if steps else 0.01


def inject(events: List, kind: str, at: int, length: int, seed: int = 0) -> List:
    """A copy of events with poses at..at+length-1 (pose order) altered and labeled ``kind``."""
    if kind not in c.INJECTIONS:
        raise ValueError("inject: %r not in %s" % (kind, c.INJECTIONS))
    idx = [i for i, e in enumerate(events) if isinstance(e, c.CameraPose)]
    poses = [events[i] for i in idx]
    if kind == "shuffle" and length < 2:
        raise ValueError("inject: shuffle needs --length >= 2 (one pose would only duplicate a neighbour)")
    if length < 1 or at < 0 or at + length > len(poses):
        raise ValueError("inject: poses %d..%d outside the recording's %d poses"
                         % (at, at + length - 1, len(poses)))
    rng = random.Random(seed)
    sigma_pos = BLUR_POS_FRACTION * _median_step(poses)
    out = list(events)
    for j in range(length):
        k = at + j
        p = poses[k]
        if kind == "shuffle":
            # Alternate ahead / behind so neighbours inside the stretch are ~2K frames apart.
            src = k + length if j % 2 == 0 else k - length
            if not 0 <= src < len(poses):  # flipping side would make a smooth run, not a jump
                raise ValueError("inject: shuffle at %d needs %d poses before and after the stretch"
                                 % (at, length))
            geo = {f: getattr(poses[src], f) for f in _GEOMETRY}
            new = replace(p, injected="shuffle", **geo)
        else:
            axis = [rng.gauss(0, 1) for _ in range(3)]
            n = math.sqrt(sum(a * a for a in axis)) or 1.0
            half = math.radians(rng.gauss(0, BLUR_ROT_DEG)) / 2
            dq = (math.cos(half),) + tuple(math.sin(half) * a / n for a in axis)
            qw, qx, qy, qz = _qmul((p.qw, p.qx, p.qy, p.qz), dq)
            # Jitter the camera centre, then recompute t = -R C so rotating does not also
            # swing the centre around the world origin.
            cx, cy, cz = (v + rng.gauss(0, sigma_pos) for v in c.camera_center(p))
            R = _rotmat((qw, qx, qy, qz))
            t = [-(R[r][0] * cx + R[r][1] * cy + R[r][2] * cz) for r in range(3)]
            new = replace(p, qw=qw, qx=qx, qy=qy, qz=qz, tx=t[0], ty=t[1], tz=t[2],
                          confidence=p.confidence * BLUR_CONF_SCALE, injected="blur")
        out[idx[k]] = c.validate(c.CameraPose, asdict(new))  # renormalizes the quaternion
    return out


# ---------------------------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------------------------

class Control:
    """Pause flag shared by the send loop, the SIGUSR1 handler and the TTY key reader."""

    def __init__(self):
        self.paused = False
        self.stopped = False
        self._wake = threading.Event()

    def toggle(self, *_):
        self.paused = not self.paused
        self._wake.set()

    def stop(self, *_):
        self.stopped = True
        self._wake.set()

    def wait(self, timeout: float) -> None:
        """Sleep up to timeout, returning early on toggle / stop."""
        self._wake.wait(timeout)
        self._wake.clear()


def watch_keys(ctrl: Control) -> Optional[Callable[[], None]]:
    """p toggles pause, q stops; only for a TTY stdin of a foreground job. Returns a restore fn."""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        if not os.isatty(fd) or os.getpgrp() != os.tcgetpgrp(fd):
            return None
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (ImportError, OSError, ValueError, AttributeError):
        return None

    def read():
        while not ctrl.stopped:
            ch = os.read(fd, 1)
            if not ch:
                return
            if ch in (b"p", b"P"):
                ctrl.toggle()
            elif ch in (b"q", b"Q"):
                ctrl.stop()

    threading.Thread(target=read, name="replay-keys", daemon=True).start()
    return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# ---------------------------------------------------------------------------------------------
# Send loop
# ---------------------------------------------------------------------------------------------

class Status:
    """One-line progress on stderr: rewritten in place on a terminal, periodic lines otherwise."""

    def __init__(self, job_id: str, total: int, rate: float, stream=None):
        self.stream = stream or sys.stderr
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.job_id, self.total, self.rate = job_id, total, rate
        self.counts = {"pose": 0, "training": 0, "injected": 0}
        self.sent = 0
        self.passes = 1
        self.dropped = 0
        self._last = 0.0
        self._last_paused = None

    def line(self, paused: bool) -> str:
        return ("[replay] %s  %d/%d sent (pose %d, training %d)  injected %d  dropped %d  "
                "%.1f ev/s  pass %d%s" % (self.job_id, self.sent, self.total * self.passes,
                                          self.counts["pose"], self.counts["training"],
                                          self.counts["injected"], self.dropped, self.rate,
                                          self.passes, "  PAUSED" if paused else ""))

    def show(self, paused: bool, force: bool = False) -> None:
        now = time.monotonic()
        changed = paused != self._last_paused
        if self.tty:
            if force or changed or now - self._last >= 0.1:
                self.stream.write("\r\033[K" + self.line(paused))
                self.stream.flush()
                self._last = now
        elif force or changed or now - self._last >= STATUS_EVERY_S:
            self.stream.write(self.line(paused) + "\n")
            self.stream.flush()
            self._last = now
        self._last_paused = paused

    def done(self) -> None:
        self.show(False, force=True)
        if self.tty:
            self.stream.write("\n")


def run(events: List, sender, job_id: str, rate: float = c.REPLAY_RATE_HZ, loop: bool = False,
        ctrl: Optional[Control] = None, status: Optional[Status] = None,
        keep_source: bool = False, max_events: Optional[int] = None) -> int:
    """Send events at ``rate`` per second until the end (or forever with loop); returns the
    number handed to the sender. Pausing stops sending; resuming does not burst to catch up."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    ctrl = ctrl or Control()
    status = status or Status(job_id, len(events), rate)
    interval = 1.0 / rate
    next_t = time.monotonic()
    sent = 0
    while True:
        for e in events:
            while True:  # wait for this event's slot, holding while paused
                if ctrl.stopped or (max_events is not None and sent >= max_events):
                    return sent
                now = time.monotonic()
                if ctrl.paused:
                    status.show(True)
                    ctrl.wait(0.2)
                    next_t = time.monotonic()  # resume at the steady pace, no catch-up burst
                    continue
                if now >= next_t:
                    break
                ctrl.wait(min(next_t - now, 0.2))
            changes = dict(job_id=job_id, seq=sender.next_seq(job_id), ts=time.time(),
                           dropped=sender.dropped(job_id))
            is_pose = isinstance(e, c.CameraPose)
            if is_pose and not keep_source:
                changes["source"] = "replay"
            sender.send(replace(e, **changes))
            sent += 1
            status.sent = sent
            status.counts["pose" if is_pose else "training"] += 1
            status.counts["injected"] += e.injected is not None
            status.dropped = sender.dropped(job_id)
            status.show(False)
            next_t = max(next_t + interval, time.monotonic() - interval)  # no backlog burst
        if not loop or not events:
            return sent
        status.passes += 1


# ---------------------------------------------------------------------------------------------
# Database reset
# ---------------------------------------------------------------------------------------------

def reset_job(job_id: str, dsn: str = c.DB_DSN) -> Dict[str, int]:
    """Delete job_id's rows; {table: rows deleted}, absent tables skipped."""
    import psycopg  # lazy: replay without --reset needs no database driver

    deleted = {}
    with psycopg.connect(dsn, connect_timeout=5) as conn:  # commits on clean exit
        for table in RESET_TABLES:
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                continue
            # table names come from RESET_TABLES, never from input
            deleted[table] = conn.execute("DELETE FROM %s WHERE job_id = %%s" % table, (job_id,)).rowcount
    return deleted


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", help="recordings/<name>.jsonl from recording.py export")
    ap.add_argument("--rate", type=float, default=c.REPLAY_RATE_HZ, help="events per second")
    ap.add_argument("--job-id", help="job id to stream as (default: replay-<timestamp>)")
    ap.add_argument("--inject", choices=c.INJECTIONS, help="bad-pose ground truth to insert")
    ap.add_argument("--at", type=int, help="first injected pose, counted in pose events from 0")
    ap.add_argument("--length", type=int, default=5, help="number of injected poses (K)")
    ap.add_argument("--seed", type=int, default=0, help="blur jitter seed")
    ap.add_argument("--reset", action="store_true",
                    help="delete --job-id's rows from the database first (alone: reset and exit)")
    ap.add_argument("--loop", action="store_true", help="replay again from the start until stopped")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep recorded pose sources (map/reloc rows are then not scored)")
    a = ap.parse_args(argv)

    if a.rate <= 0:
        ap.error("--rate must be > 0")
    if a.inject and a.at is None:
        ap.error("--inject needs --at")
    if a.reset and not a.job_id:
        ap.error("--reset needs --job-id (a fresh replay-<timestamp> id has no rows)")
    if not a.recording and not a.reset:
        ap.error("--recording is required")

    if a.recording and not os.path.isfile(a.recording):
        ap.error("--recording %s not found" % a.recording)
    events = []
    if a.recording:  # load and inject before --reset so a bad argument never wipes rows
        events = load(a.recording)
        if not events:
            print("[replay] %s has no valid events" % a.recording, file=sys.stderr)
            return 1
        if a.inject:
            try:
                events = inject(events, a.inject, a.at, a.length, a.seed)
            except ValueError as e:
                ap.error(str(e))

    if a.reset:
        try:
            deleted = reset_job(a.job_id)
        except ImportError:
            print("[replay] --reset needs psycopg (pip install 'psycopg[binary]')", file=sys.stderr)
            return 1
        except Exception as e:  # psycopg.OperationalError etc.; driver imported lazily
            print("[replay] --reset failed: %s" % e, file=sys.stderr)
            return 1
        print("[replay] reset %s: %s" % (a.job_id, ", ".join("%s %d" % kv for kv in deleted.items())
                                         or "no tables yet"), file=sys.stderr)
        if not a.recording:
            return 0

    from hypersplat.streaming.producer import Sender  # lazy: kafka only when actually sending

    job_id = a.job_id or time.strftime("replay-%Y%m%d-%H%M%S-") + "%d" % (os.getpid() % 1000)
    sender = Sender()
    ctrl = Control()
    signal.signal(signal.SIGUSR1, ctrl.toggle)
    signal.signal(signal.SIGTERM, ctrl.stop)
    restore = watch_keys(ctrl)
    status = Status(job_id, len(events), a.rate)
    print("[replay] %s -> %s as %s at %.1f ev/s (pid %d: kill -USR1 to pause%s)"
          % (a.recording, sender.bootstrap, job_id, a.rate, os.getpid(),
             ", or press p" if restore else ""), file=sys.stderr)
    try:
        run(events, sender, job_id, a.rate, loop=a.loop, ctrl=ctrl, status=status,
            keep_source=a.keep_source)
    except KeyboardInterrupt:
        pass
    finally:
        if restore:
            restore()
        sender.close()
        status.dropped = sender.dropped(job_id)
        status.done()
    return 0


if __name__ == "__main__":
    sys.exit(main())
