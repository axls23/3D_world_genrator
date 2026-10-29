"""Non-blocking Kafka sender shared by every producer (rows 2, 4, 5; constraint "training
speed comes first").

    hook / replay thread --send(event)--> bounded queue --daemon thread--> KafkaProducer

send() only does put_nowait, so the trainer loop and the ACE reader / ring threads never wait
on Kafka. A full queue, a broker that is down or an event that fails to encode is dropped and
counted per job; the count rides on that job's next events as ``dropped`` (dashboard counter).
Stdlib + kafka-python (pure Python), py3.8-compatible like contract.py.
"""

import atexit
import logging
import queue
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from . import contract as c

logger = logging.getLogger(__name__)

RECONNECT_S = 5.0  # broker down: wait this long before trying to connect again


def _kafka_producer(bootstrap: str):
    from kafka import KafkaProducer  # lazy: importing the contract must not need kafka

    return KafkaProducer(bootstrap_servers=bootstrap, **c.PRODUCER_CONFIG)


class Sender:
    """One per process (get_sender()); thread-safe."""

    def __init__(self, bootstrap: str = c.KAFKA_BOOTSTRAP, queue_max: int = c.PRODUCER_QUEUE_MAX,
                 producer_factory: Optional[Callable[[str], object]] = None):
        self.bootstrap = bootstrap
        self._factory = producer_factory or _kafka_producer
        self._q = queue.Queue(maxsize=queue_max)
        self._lock = threading.Lock()
        self._seq = defaultdict(int)       # job_id -> next seq
        self._dropped = defaultdict(int)   # job_id -> events dropped so far
        self.sent = 0
        self._producer = None
        self._retry_at = 0.0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hypersplat-kafka-sender", daemon=True)
        self._thread.start()

    # -- caller side (hot path) ----------------------------------------------------------------
    def next_seq(self, job_id: str) -> int:
        with self._lock:
            s = self._seq[job_id]
            self._seq[job_id] = s + 1
            return s

    def drop(self, job_id: str, reason: str = "") -> None:
        with self._lock:
            self._dropped[job_id] += 1
            n = self._dropped[job_id]
        if n == 1 or n % 100 == 0:  # first and then every 100th, never per event
            logger.warning("[streaming] dropped %d event(s) for %s (%s)", n, job_id, reason)

    def dropped(self, job_id: str) -> int:
        with self._lock:
            return self._dropped[job_id]

    def stats(self, job_id: Optional[str] = None) -> dict:
        """What this sender is losing, for logs and dashboards: events sent, dropped (one job, or all),
        how full the queue is, and whether the broker connection is up."""
        with self._lock:
            dropped = self._dropped[job_id] if job_id is not None else sum(self._dropped.values())
        return {"sent": self.sent, "dropped": dropped, "queue_depth": self._q.qsize(),
                "queue_max": self._q.maxsize, "broker_up": self._producer is not None}

    def send(self, event: c.Event) -> bool:
        """Queue one event; False if it was dropped. Never blocks, never raises."""
        if self._closed.is_set():
            self.drop(event.job_id, "closed")
            return False
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.drop(event.job_id, "queue full")
            return False

    # -- sender thread -------------------------------------------------------------------------
    def _connect(self):
        if self._producer is None and time.time() >= self._retry_at:
            try:
                self._producer = self._factory(self.bootstrap)
            except Exception as e:
                self._retry_at = time.time() + RECONNECT_S
                logger.warning("[streaming] Kafka at %s unreachable (%s); dropping events", self.bootstrap, e)
        return self._producer

    def _run(self):
        while not (self._closed.is_set() and self._q.empty()):
            try:
                event = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._deliver(event)
            finally:
                self._q.task_done()

    def _deliver(self, event: c.Event) -> None:
        try:
            topic, key, value = c.encode(event)
        except c.InvalidEvent as e:
            self.drop(event.job_id, str(e))
            return
        producer = self._connect()
        if producer is None:
            self.drop(event.job_id, "broker down")
            return
        try:
            producer.send(topic, key=key, value=value)
            self.sent += 1
        except Exception as e:
            self.drop(event.job_id, "send failed: %s" % e)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued events are handed to Kafka (end of run, tests)."""
        deadline = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < deadline:
            time.sleep(0.01)
        if self._producer is not None:
            try:
                self._producer.flush(timeout=max(0.0, deadline - time.time()))
            except Exception as e:
                logger.warning("[streaming] flush failed: %s", e)

    def close(self, timeout: float = 5.0) -> None:
        self.flush(timeout)
        self._closed.set()
        self._thread.join(timeout=1.0)
        if self._producer is not None:
            try:
                self._producer.close(timeout=1.0)
            except Exception:
                pass


class FakeSender(Sender):
    """Collects (topic, key, value) in memory instead of talking to Kafka (tests)."""

    def __init__(self, queue_max: int = c.PRODUCER_QUEUE_MAX):
        self.messages: List[Tuple[str, bytes, bytes]] = []
        outer = self

        class _Producer:
            def send(self, topic, key=None, value=None):
                outer.messages.append((topic, key, value))

            def flush(self, timeout=None):
                pass

            def close(self, timeout=None):
                pass

        super().__init__("fake", queue_max, lambda _: _Producer())

    def events(self) -> List[c.Event]:
        return [c.decode(v) for _, _, v in self.messages]


_SENDER: Optional[Sender] = None
_SENDER_LOCK = threading.Lock()


def get_sender() -> Sender:
    """The process-wide Sender, created on first use and flushed at exit."""
    global _SENDER
    with _SENDER_LOCK:
        if _SENDER is None:
            _SENDER = Sender()
            atexit.register(_SENDER.close, 2.0)
        return _SENDER
