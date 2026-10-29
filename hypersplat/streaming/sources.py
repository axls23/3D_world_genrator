"""File-following pose source for the batch / WSL path (Policy & Process Map rows 2, 3, 4;
constraint "keep the old path as a fallback").

ACE keeps writing poses_final.txt exactly as before; this process follows that file and sends
one CameraPose (source="file") per new line, so the shared-folder hand-off and Kafka run side by
side until Kafka has carried one full reconstruction cleanly.

    python -m hypersplat.streaming.sources tail --pose-file <dir>/poses_final.txt --job-id X

Handles a file that does not exist yet, partial last lines, truncation and rewrites (a refit
rewrites the whole file): after a rewrite only lines not sent before go out, so an unchanged pose
is not duplicated while a moved one is. py3.8-compatible (runs in the ACE env too).
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional, Set

from . import contract as c

logger = logging.getLogger(__name__)

_PREFIX = 256  # bytes compared to tell an in-place rewrite from an append


class PoseFileTail(object):
    """Incremental reader of one ACE pose file. poll() returns the complete new lines."""

    def __init__(self, path: str):
        self.path = path
        self.pos = 0
        self.ino = None         # type: Optional[int]
        self.prefix = b""
        self.seen = set()       # type: Set[str]
        self.rewrites = 0

    def _reset(self) -> None:
        self.pos, self.prefix = 0, b""
        self.rewrites += 1

    def poll(self, final: bool = False) -> List[str]:
        """New complete lines since the last call; ``final`` also returns an unterminated tail."""
        try:
            f = open(self.path, "rb")
        except OSError:
            return []  # not written yet
        with f:
            st = os.fstat(f.fileno())
            if self.ino is not None and (st.st_ino != self.ino or st.st_size < self.pos):
                self._reset()  # replaced (new inode) or truncated
            elif self.prefix and f.read(len(self.prefix)) != self.prefix:
                self._reset()  # rewritten in place with different content
            self.ino = st.st_ino
            if len(self.prefix) < _PREFIX:
                f.seek(0)
                self.prefix = f.read(_PREFIX)
            f.seek(self.pos)
            data = f.read()
        end = len(data) if final else data.rfind(b"\n") + 1
        self.pos += end
        out = []
        for raw in data[:end].decode("utf-8", "replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line in self.seen:
                continue
            self.seen.add(line)
            out.append(line)
        return out


def send_lines(lines: List[str], job_id: str, sender) -> int:
    """One CameraPose per pose line; bad lines are dropped and counted. Returns events queued."""
    n = 0
    for line in lines:
        try:
            ev = c.pose_from_ace_line(line, job_id, sender.next_seq(job_id), source="file",
                                      dropped=sender.dropped(job_id))
        except c.InvalidEvent as e:
            sender.drop(job_id, str(e))
            continue
        n += sender.send(ev)
    return n


def tail(pose_file: str, job_id: str, sender, poll_s: float = 0.5, once: bool = False,
         idle_exit_s: float = 0.0) -> int:
    """Follow ``pose_file`` until Ctrl+C (or once / after ``idle_exit_s`` without new lines)."""
    t = PoseFileTail(pose_file)
    total, last_new = 0, time.monotonic()
    try:
        while True:
            lines = t.poll(final=once)
            if lines:
                total += send_lines(lines, job_id, sender)
                last_new = time.monotonic()
            if once or (idle_exit_s > 0 and time.monotonic() - last_new > idle_exit_s):
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        pass
    logger.info("[streaming] %s: %d pose(s) queued, %d rewrite(s), %d dropped",
                pose_file, total, t.rewrites, sender.dropped(job_id))
    return total


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.sources", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    tp = sub.add_parser("tail", help="follow an ACE poses_final.txt and send each new pose")
    tp.add_argument("--pose-file", required=True)
    tp.add_argument("--job-id", default=os.environ.get("HYPERSPLAT_JOB_ID"),
                    help="default: $HYPERSPLAT_JOB_ID")
    tp.add_argument("--poll", type=float, default=0.5, help="seconds between file checks")
    tp.add_argument("--once", action="store_true", help="send what is there now and exit")
    tp.add_argument("--idle-exit", type=float, default=0.0,
                    help="exit after this many seconds without a new line (0: never)")
    args = ap.parse_args(argv)
    if args.cmd != "tail":
        ap.print_help()
        return 2
    if not args.job_id:
        ap.error("--job-id (or HYPERSPLAT_JOB_ID) is required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .producer import get_sender

    sender = get_sender()
    tail(args.pose_file, args.job_id, sender, args.poll, args.once, args.idle_exit)
    sender.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
