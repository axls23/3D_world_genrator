"""Create and list the three streaming topics (Policy & Process Map rows 6 and 19).

    python -m hypersplat.streaming.topics create [--wait 60]   idempotent; start-order step 2
    python -m hypersplat.streaming.topics list [--all]         topics with message counts
    python -m hypersplat.streaming.topics delete --yes         reset (runbook), contract topics only

``list`` is the "Kafka topic list with message counts" evidence: a topic's message count is the
sum of its partitions' end offsets (every message ever written), with the retained count
(end - beginning offsets) beside it, since old segments age out after Topic.retention_ms.
Names, partitions and retention come from the contract; HYPERSPLAT_TOPIC_PREFIX lets tests and
parallel setups share one broker. Stdlib + kafka-python, imported lazily.
"""

import argparse
import socket
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Sequence

from . import contract as c

REPLICATION_FACTOR = 1  # one KRaft node (docker-compose.yml, ops/infra.sh)
CLIENT_ID = "hypersplat-topics"


class TopicCount(NamedTuple):
    name: str
    partitions: int
    messages: int   # sum of end offsets
    retained: int   # sum of (end - beginning) offsets


def _host_port(bootstrap: str):
    """(host, port) of the first bootstrap server; kafka-python's default port is 9092."""
    first = bootstrap.split(",")[0].strip()
    if first.startswith("["):  # [v6addr] or [v6addr]:port
        host, _, rest = first[1:].partition("]")
        return host, int(rest[1:]) if rest.startswith(":") else 9092
    if first.count(":") == 1:
        host, port = first.split(":")
        return host or "localhost", int(port)
    return first or "localhost", 9092  # bare host name or bare IPv6 address


def broker_reachable(bootstrap: str = c.KAFKA_BOOTSTRAP, timeout: float = 1.0) -> bool:
    """TCP connect to the first bootstrap server; cheap pre-check before kafka-python's retries."""
    try:
        with socket.create_connection(_host_port(bootstrap), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def wait_for_broker(bootstrap: str, timeout: float) -> None:
    """Block until the broker answers an admin request (kafka-init / up.sh step 2).

    An open port is not enough: Docker's port proxy accepts connections before Kafka listens.
    """
    deadline = time.time() + timeout
    last = None
    while True:
        if broker_reachable(bootstrap):
            try:
                _admin(bootstrap).close()
                return
            except Exception as e:  # NoBrokersAvailable while the broker is still starting
                last = e
        if time.time() >= deadline:
            raise SystemExit("Kafka at %s not ready after %.0f s (%s)" % (bootstrap, timeout, last))
        time.sleep(1.0)


def _admin(bootstrap: str):
    from kafka.admin import KafkaAdminClient  # lazy: the contract must import without kafka

    return KafkaAdminClient(bootstrap_servers=bootstrap, client_id=CLIENT_ID,
                            request_timeout_ms=10000)


def create_topics(bootstrap: str = c.KAFKA_BOOTSTRAP,
                  topics: Sequence[c.Topic] = c.TOPICS) -> Dict[str, str]:
    """Create missing topics and bring existing ones to the contract; {name: what happened}.

    An existing topic (e.g. auto-created by a client on a broker that allows it) gets the
    contract's retention.ms set again; a partition mismatch is only reported, since shrinking
    partitions is impossible and growing them would reorder a job's events.
    """
    from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = _admin(bootstrap)
    try:
        existing = set(admin.list_topics())
        status = {}
        for t in topics:
            if t.name not in existing:
                new = NewTopic(name=t.name, num_partitions=t.partitions,
                               replication_factor=REPLICATION_FACTOR,
                               topic_configs={"retention.ms": str(t.retention_ms)})
                try:  # one call per topic so a concurrent creator only costs that topic
                    admin.create_topics([new])
                    status[t.name] = "created"
                    continue
                except TopicAlreadyExistsError:
                    pass
            admin.alter_configs([ConfigResource(ConfigResourceType.TOPIC, t.name,
                                                configs={"retention.ms": str(t.retention_ms)})])
            n = len(admin.describe_topics([t.name])[0]["partitions"])
            status[t.name] = "exists" if n == t.partitions else (
                "exists, WARNING %d partitions, contract says %d" % (n, t.partitions))
        return status
    finally:
        admin.close()


def delete_topics(bootstrap: str = c.KAFKA_BOOTSTRAP, topics: Sequence[c.Topic] = c.TOPICS,
                  timeout: float = 30.0) -> List[str]:
    """Delete the given topics if present and wait until the broker no longer lists them
    (deletion is asynchronous; a create right after would otherwise see them as existing)."""
    from kafka.errors import UnknownTopicOrPartitionError

    admin = _admin(bootstrap)
    try:
        names = [t.name for t in topics if t.name in set(admin.list_topics())]
        if names:
            try:
                admin.delete_topics(names)
            except UnknownTopicOrPartitionError:  # a concurrent delete got there first
                pass
        deadline = time.time() + timeout
        while set(names) & set(admin.list_topics()):
            if time.time() >= deadline:
                raise RuntimeError("topics still listed %.0f s after delete: %s" % (timeout, names))
            time.sleep(0.5)
        return names
    finally:
        admin.close()


def topic_counts(bootstrap: str = c.KAFKA_BOOTSTRAP, names: Optional[Sequence[str]] = None,
                 wait_s: float = 0.0) -> List[TopicCount]:
    """Partition count and message counts per topic; a missing topic reports -1 partitions.

    ``names=None`` lists every non-internal topic on the broker. ``wait_s`` gives topics that
    were just created time to show up in the metadata before they are reported missing.
    """
    from kafka import KafkaConsumer, TopicPartition

    consumer = KafkaConsumer(bootstrap_servers=bootstrap, client_id=CLIENT_ID, group_id=None,
                             enable_auto_commit=False, request_timeout_ms=10000)
    try:
        if names is None:
            names = sorted(n for n in consumer.topics() if not n.startswith("__"))
        rows = []
        deadline = time.time() + wait_s
        for name in names:
            parts = consumer.partitions_for_topic(name)
            while not parts and time.time() < deadline:
                time.sleep(0.2)
                consumer.topics()  # forces a metadata refresh
                parts = consumer.partitions_for_topic(name)
            if not parts:
                rows.append(TopicCount(name, -1, 0, 0))
                continue
            tps = [TopicPartition(name, p) for p in sorted(parts)]
            end = consumer.end_offsets(tps)
            begin = consumer.beginning_offsets(tps)
            rows.append(TopicCount(name, len(tps), sum(end.values()),
                                   sum(end[tp] - begin[tp] for tp in tps)))
        return rows
    finally:
        consumer.close()


def format_table(rows: Sequence[TopicCount], retention: Optional[Dict[str, int]] = None) -> str:
    retention = retention or {}
    width = max([len("topic")] + [len(r.name) for r in rows])
    lines = ["%-*s  %10s  %10s  %10s  %9s" % (width, "topic", "partitions", "messages",
                                             "retained", "retention")]
    for r in rows:
        ret = retention.get(r.name)
        lines.append("%-*s  %10s  %10d  %10d  %9s" % (
            width, r.name, r.partitions if r.partitions >= 0 else "missing", r.messages,
            r.retained, "%gd" % (ret / 86400000.0) if ret else "-"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.topics",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--bootstrap", default=c.KAFKA_BOOTSTRAP,
                    help="Kafka bootstrap servers (default %(default)s, env HYPERSPLAT_KAFKA)")
    ap.add_argument("--wait", type=float, default=0.0, metavar="S",
                    help="wait up to S seconds for the broker before giving up")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="create missing contract topics, reapply retention (idempotent)")
    ls = sub.add_parser("list", help="contract topics with message counts")
    ls.add_argument("--all", action="store_true", help="every topic on the broker")
    rm = sub.add_parser("delete", help="delete the contract topics (reset)")
    rm.add_argument("--yes", action="store_true", help="required; deletes all their messages")
    args = ap.parse_args(argv)

    if args.wait > 0:
        wait_for_broker(args.bootstrap, args.wait)
    elif not broker_reachable(args.bootstrap):
        print("Kafka at %s is not reachable (start it: ops/up.sh or ops/infra.sh up)"
              % args.bootstrap, file=sys.stderr)
        return 2

    if args.cmd == "create":
        for name, what in create_topics(args.bootstrap).items():
            print("%-8s %s" % (what, name))
    elif args.cmd == "delete":
        if not args.yes:
            print("refusing to delete %s without --yes" % ", ".join(t.name for t in c.TOPICS),
                  file=sys.stderr)
            return 1
        for name in delete_topics(args.bootstrap):
            print("deleted  %s" % name)
    if args.cmd in ("create", "list"):
        names = None if getattr(args, "all", False) else [t.name for t in c.TOPICS]
        rows = topic_counts(args.bootstrap, names, wait_s=10.0 if args.cmd == "create" else 0.0)
        print(format_table(rows, {t.name: t.retention_ms for t in c.TOPICS}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
