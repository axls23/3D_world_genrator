"""Arrow Flight endpoint of a streaming session: external cameras in, live state out.

Server (started by the session for a ``flight://host:port`` source, or alongside any other
source when a Flight port is configured):

    do_put  descriptor path ["frames"]   FRAME_SCHEMA batches (t, jpeg) from a camera
    do_get  ticket b"views"              current views (poses, intrinsics, weights; no pixels)
    do_get  ticket b"stats"              one-row table of pipeline counters

Producer - stream a local camera or a video to a remote reconstruction:

    python -m hypersplat.stream.flight push 0 grpc://gpu-box:8815 --fps 10
    python -m hypersplat.stream.flight push drone.mp4 grpc://localhost:8815
    python -m hypersplat.stream.flight get grpc://gpu-box:8815 stats

Frames are JPEG-encoded at the producer (``--max_side``), so a 960 px, 10 fps camera costs
~1 MB/s of bandwidth. Only the newest frame is kept when the reconstruction is busy.
"""

import argparse
import json
import logging
import threading
import time
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.flight as flight

from .schema import FRAME_SCHEMA
from .source import FrameSource, QueueFrameSource, decode_jpeg, encode_jpeg

logger = logging.getLogger(__name__)


def flight_location(spec: str) -> str:
    """flight://host:port or grpc://host:port -> grpc://host:port (what pyarrow expects)."""
    return "grpc://" + spec.split("://", 1)[1] if "://" in spec else "grpc://" + spec


class FrameFlightServer(flight.FlightServerBase):
    def __init__(self, location: str, frames: Optional[QueueFrameSource] = None,
                 views: Optional[Callable[[], pa.Table]] = None,
                 stats: Optional[Callable[[], dict]] = None):
        super().__init__(location)
        self.frames = frames
        self._views = views
        self._stats = stats
        self.producers = 0

    def do_put(self, context, descriptor, reader, writer):
        path = [p.decode() if isinstance(p, bytes) else p for p in (descriptor.path or [])]
        if path != ["frames"] or self.frames is None:
            raise flight.FlightServerError("this server accepts do_put on path ['frames'] only")
        self.producers += 1
        try:
            for chunk in reader:
                batch = chunk.data
                for t, data in zip(batch.column("t").to_pylist(), batch.column("jpeg").to_pylist()):
                    try:
                        self.frames.push(t, decode_jpeg(data))
                    except ValueError:
                        logger.warning("[Flight] dropped an undecodable frame")
        finally:
            self.producers -= 1

    def do_get(self, context, ticket):
        name = ticket.ticket.decode()
        if name == "views" and self._views is not None:
            table = self._views()
            if "jpeg" in table.column_names:
                table = table.drop_columns(["jpeg"])
            return flight.RecordBatchStream(table)
        if name == "stats" and self._stats is not None:
            stats = self._stats()
            return flight.RecordBatchStream(pa.Table.from_pylist([{k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                                                     for k, v in stats.items()}]))
        raise flight.FlightServerError(f"unknown ticket {name!r} (views | stats)")

    def list_flights(self, context, criteria):
        for name in ("views", "stats"):
            yield flight.FlightInfo(pa.schema([]), flight.FlightDescriptor.for_path(name),
                                    [flight.FlightEndpoint(name, [])], -1, -1)

    def serve_in_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve, name="flight-server", daemon=True)
        t.start()
        return t


def push(source_spec: str, location: str, fps: float = 10.0, max_side: int = 960,
         quality: int = 85, batch_frames: int = 1) -> int:
    """Stream frames from a local source to a FrameFlightServer. Returns frames sent."""
    client = flight.connect(flight_location(location))
    writer, _ = client.do_put(flight.FlightDescriptor.for_path("frames"), FRAME_SCHEMA)
    src = FrameSource(source_spec, max_side=max_side, pace="realtime")
    interval = 1.0 / fps if fps > 0 else 0.0
    sent, last, pending = 0, 0.0, []
    try:
        for frame in src:
            now = time.time()
            if interval and now - last < interval:
                continue
            last = now
            pending.append({"t": frame.t, "jpeg": encode_jpeg(frame.image, quality)})
            if len(pending) >= batch_frames:
                writer.write_batch(pa.RecordBatch.from_pylist(pending, schema=FRAME_SCHEMA))
                sent += len(pending)
                pending = []
    except KeyboardInterrupt:
        pass
    finally:
        if pending:
            writer.write_batch(pa.RecordBatch.from_pylist(pending, schema=FRAME_SCHEMA))
            sent += len(pending)
        writer.close()
        client.close()
    return sent


def get(location: str, ticket: str) -> pa.Table:
    client = flight.connect(flight_location(location))
    try:
        return client.do_get(flight.Ticket(ticket.encode())).read_all()
    finally:
        client.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="stream a camera/video to a reconstruction server")
    p.add_argument("source", help="camera index, /dev/videoN, URL or video file")
    p.add_argument("location", help="grpc://host:port of the reconstruction")
    p.add_argument("--fps", type=float, default=10.0, help="frames per second to send (0 = all)")
    p.add_argument("--max_side", type=int, default=960)
    p.add_argument("--quality", type=int, default=85, help="JPEG quality")
    g = sub.add_parser("get", help="read live state from a reconstruction server")
    g.add_argument("location")
    g.add_argument("ticket", choices=["views", "stats"])
    args = ap.parse_args()
    if args.cmd == "push":
        n = push(args.source, args.location, args.fps, args.max_side, args.quality)
        logger.info(f"sent {n} frames")
    else:
        print(get(args.location, args.ticket).to_pandas().to_string())


if __name__ == "__main__":
    main()
