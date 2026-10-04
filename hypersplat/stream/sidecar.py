"""Client for the ACE localization sidecar (scripts/acezero/stream_sidecar.py).

The sidecar runs in the ACE-Zero env; this side spawns it and exchanges Arrow IPC batches
over its stdin/stdout. Events are delivered on a reader thread to ``on_event(kind, rows)``
where rows are EVENT_SCHEMA dicts with w2c as a 4x4 array and points/pix as (N, 3)/(N, 2).

Commands are queued and written by their own thread, so sending never blocks the caller.
Handlers on the reader thread send commands (the first map localizes every pending frame),
and pose events are ~30 KB each: if the reader blocked on a full stdin pipe while the
sidecar blocked on a full stdout pipe that only the reader drains, both would hang.
"""

import logging
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .schema import COMMAND_SCHEMA, IpcWriter, command_batch, ipc_batches, parse_meta

logger = logging.getLogger(__name__)

ACEZERO_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "acezero"
SIDECAR_SCRIPT = ACEZERO_ROOT / "stream_sidecar.py"


def decode_event_rows(batch) -> List[Dict]:
    rows = batch.to_pylist()
    for r in rows:
        r["w2c"] = np.asarray(r["w2c"], dtype=np.float64).reshape(4, 4) if r["w2c"] is not None else None
        r["points"] = np.asarray(r["points"] or [], dtype=np.float32).reshape(-1, 3)
        r["pix"] = np.asarray(r["pix"] or [], dtype=np.float32).reshape(-1, 2)
        r["meta"] = parse_meta(r["meta"])
    return rows


class AceSidecar:
    def __init__(self, work_dir: Path, on_event: Callable[[str, List[Dict]], None],
                 hypotheses: int = 64, threshold: float = 10.0, max_points: int = 1500,
                 python: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        from hypersplat.pipeline.wrappers.perception import acezero_python_env

        default_python, default_env = acezero_python_env(ACEZERO_ROOT)
        # The trainer's allocator options (e.g. expandable_segments) are for its torch; the ACE
        # env runs torch 2.0, which rejects unknown options and would crash on start
        default_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        cmd = [python or default_python, "-u", str(SIDECAR_SCRIPT), "--work_dir", str(work_dir),
               "--hypotheses", str(hypotheses), "--threshold", str(threshold),
               "--max_points", str(max_points)]
        # stderr is inherited: ACE progress shows up in the trainer's log
        # Own session: close() can take down the sidecar together with a running ACE job
        self.proc = subprocess.Popen(cmd, cwd=str(ACEZERO_ROOT), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env or default_env,
                                     start_new_session=True)
        self.on_event = on_event
        self._writer = IpcWriter(self.proc.stdin, COMMAND_SCHEMA)
        self._outbox: "queue.Queue" = queue.Queue()  # command batches; None ends the writer
        self.ready = threading.Event()
        self._reader = threading.Thread(target=self._read, name="ace-sidecar-reader", daemon=True)
        self._reader.start()
        self._writer_thread = threading.Thread(target=self._write, name="ace-sidecar-writer", daemon=True)
        self._writer_thread.start()

    def _read(self):
        try:
            for batch in ipc_batches(self.proc.stdout):
                rows = decode_event_rows(batch)
                if not rows:
                    continue
                kind = rows[0]["kind"]
                if kind == "ready":
                    self.ready.set()
                    continue
                try:
                    self.on_event(kind, rows)
                except Exception:
                    logger.exception("sidecar event handler failed")
        except Exception as e:  # pipe torn down (shutdown or crash)
            if self.proc.poll() is None:
                logger.warning(f"[Stream] sidecar stream closed: {e}")
        finally:
            self.ready.set()
            self.on_event("exit", [{"kind": "exit", "meta": {"returncode": self.proc.poll()}}])

    def _write(self):
        while True:
            batch = self._outbox.get()
            if batch is None:
                return
            try:
                self._writer.write(batch)
            except Exception as e:  # pipe torn down: the reader reports the exit
                if self.proc.poll() is None:
                    logger.warning(f"[Stream] sidecar command stream closed: {e}")
                return

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _send(self, cmd: str, frames: Iterable[str] = (), args: Iterable[str] = (), **params):
        if not self.alive:
            raise RuntimeError(f"ACE sidecar exited with code {self.proc.returncode}")
        self._outbox.put(command_batch(cmd, frames, args, **params))

    def bootstrap(self, frames: List[str], args: List[str], focal: float = -1.0):
        self._send("bootstrap", frames, args, focal=focal)

    def localize(self, frame: str):
        self._send("localize", [frame])

    def refit(self, frames: List[str], iterations: int, conf_threshold: float, args: List[str]):
        self._send("refit", frames, args, iterations=iterations, conf_threshold=conf_threshold)

    def load(self, head: str, pose_file: str, frames: List[str], image_resolution: int = 480):
        self._send("load", frames, head=head, pose_file=pose_file, image_resolution=image_resolution)

    def forget(self, frames: List[str]):
        if frames:
            self._send("forget", frames)

    def close(self, timeout: float = 8.0):
        if self.alive:
            try:
                self._send("shutdown")  # the sidecar terminates a running ACE job itself
                self._outbox.put(None)
                self._writer_thread.join(timeout)
                self._writer.close()
                self.proc.stdin.close()
                self.proc.wait(timeout)
            except Exception:
                pass
        # Sweep the sidecar's session: a stuck sidecar or an ACE job that outlived it
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
