"""Apache Arrow schemas shared by every hop of the streaming pipeline.

    camera / video / Flight producer --FRAME--> session --COMMAND--> ACE sidecar
                                                session <--EVENT---- ACE sidecar
    session --VIEW table--> trainer (and Flight do_get "views" for monitoring)

Frames travel as JPEG bytes (a 960 px frame is ~100 KB instead of 1.5 MB raw), poses as
fixed-size float64 lists, seed points as flat float32 lists - no base64/JSON on the hot path.
The sidecar imports this module from the ACE-Zero env, so it must stay py3.8-compatible and
depend on pyarrow only.
"""

import io
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pyarrow as pa

POSE = pa.list_(pa.float64(), 16)  # row-major 4x4

# A frame from any source. t: seconds since the source started (media time for files).
FRAME_SCHEMA = pa.schema([
    pa.field("t", pa.float64()),
    pa.field("jpeg", pa.binary()),
])

# Client -> sidecar. params: small JSON object (focal, iterations, thresholds, paths).
COMMAND_SCHEMA = pa.schema([
    pa.field("cmd", pa.string()),
    pa.field("frames", pa.list_(pa.string())),
    pa.field("args", pa.list_(pa.string())),
    pa.field("params", pa.string()),
])

# Sidecar -> client. One row per frame; a map update is one batch with a row per frame.
# w2c: world-to-camera in the ACE map (OpenCV / COLMAP convention), null if not localized.
# points: inlier scene coordinates, flat (N*3); pix: their image positions in [0, 1], flat (N*2).
EVENT_SCHEMA = pa.schema([
    pa.field("kind", pa.string()),       # ready | pose | map | error
    pa.field("version", pa.int32()),     # map version the pose belongs to
    pa.field("frame", pa.string()),
    pa.field("w2c", POSE),
    pa.field("conf", pa.float32()),
    pa.field("focal", pa.float64()),     # focal of the frame files, px
    pa.field("points", pa.list_(pa.float32())),
    pa.field("pix", pa.list_(pa.float32())),
    pa.field("meta", pa.string()),       # JSON: reason, seconds, msg
])

# Trainer-facing views: real keyframes and Difix pseudo views, in the trainer's normalized
# world frame. Built by the session per version; immutable, so safe to hand across threads.
VIEW_SCHEMA = pa.schema([
    pa.field("kid", pa.int64()),         # keyframe id (pseudo views: -1 - index)
    pa.field("name", pa.string()),
    pa.field("kind", pa.string()),       # real | pseudo
    pa.field("split", pa.string()),      # train | val
    pa.field("c2w", POSE),
    pa.field("K", pa.list_(pa.float64(), 9)),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("weight", pa.float32()),    # per-view loss weight
    pa.field("conf", pa.float32()),      # ACE registration confidence (pseudo: of the source)
    pa.field("src_kid", pa.int64()),     # pseudo views: keyframe they were derived from
    pa.field("jpeg", pa.binary()),
])


def command_batch(cmd: str, frames: Iterable[str] = (), args: Iterable[str] = (),
                  **params: Any) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(
        [{"cmd": cmd, "frames": [str(f) for f in frames], "args": [str(a) for a in args],
          "params": json.dumps(params)}], schema=COMMAND_SCHEMA)


def event_batch(rows: List[Dict[str, Any]]) -> pa.RecordBatch:
    """rows: dicts with EVENT_SCHEMA keys; missing keys are null, meta may be a dict."""
    out = []
    for r in rows:
        r = dict(r)
        if isinstance(r.get("meta"), dict):
            r["meta"] = json.dumps(r["meta"])
        out.append({f.name: r.get(f.name) for f in EVENT_SCHEMA})
    return pa.RecordBatch.from_pylist(out, schema=EVENT_SCHEMA)


class IpcWriter:
    """Arrow IPC stream on a binary file object (a pipe), flushed after every batch so the
    reader on the other end sees it immediately."""

    def __init__(self, fileobj, schema: pa.Schema):
        self._file = fileobj
        self._writer = pa.ipc.new_stream(pa.PythonFile(fileobj, mode="w"), schema)

    def write(self, batch: pa.RecordBatch) -> None:
        self._writer.write_batch(batch)
        self._file.flush()

    def close(self) -> None:
        try:
            self._writer.close()
            self._file.flush()
        except (OSError, ValueError, pa.ArrowException):
            pass


def ipc_batches(fileobj) -> Iterator[pa.RecordBatch]:
    """Batches of an Arrow IPC stream on a binary file object, until it closes."""
    reader = pa.ipc.open_stream(pa.PythonFile(fileobj, mode="r"))
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue()


def select_rows(table: pa.Table, **equals: Any) -> pa.Table:
    """Rows whose columns equal the given values, e.g. select_rows(t, kind="real", split="val")."""
    import pyarrow.compute as pc

    mask = None
    for column, value in equals.items():
        m = pc.equal(table.column(column), value)
        mask = m if mask is None else pc.and_(mask, m)
    return table if mask is None else table.filter(mask)


def parse_meta(value: Optional[str]) -> Dict[str, Any]:
    return json.loads(value) if value else {}
