"""Unit 1 infra: topics CLI, docker-compose.yml against the contract, ops/up.sh start order.

Kafka tests use throwaway topic names and skip when the broker is unreachable.
"""
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from hypersplat.streaming import contract as c
from hypersplat.streaming import topics

PKG = Path(__file__).resolve().parents[1]
COMPOSE = PKG / "docker-compose.yml"
UP_SH = PKG / "ops" / "up.sh"
STEPS = ["broker", "topics", "anomaly models", "processor", "dashboard", "producer"]


# -- topics.py, no broker ---------------------------------------------------------------------

def test_host_port_parsing():
    assert topics._host_port("localhost:9094") == ("localhost", 9094)
    assert topics._host_port("kafka:29092,other:1") == ("kafka", 29092)
    assert topics._host_port("[::1]:9094") == ("::1", 9094)
    assert topics._host_port("[::1]") == ("::1", 9092)
    assert topics._host_port("kafka") == ("kafka", 9092)


def test_format_table_marks_missing_and_retention():
    rows = [topics.TopicCount("a.poses", 1, 12, 10), topics.TopicCount("b.gone", -1, 0, 0)]
    out = topics.format_table(rows, {"a.poses": 7 * 86400000})
    lines = out.splitlines()
    assert lines[0].split() == ["topic", "partitions", "messages", "retained", "retention"]
    assert lines[1].split() == ["a.poses", "1", "12", "10", "7d"]
    assert lines[2].split() == ["b.gone", "missing", "0", "0", "-"]


def test_cli_reports_unreachable_broker(capsys):
    assert topics.main(["--bootstrap", "127.0.0.1:1", "list"]) == 2
    assert "not reachable" in capsys.readouterr().err


def test_delete_needs_yes():
    if not topics.broker_reachable():
        pytest.skip("Kafka at %s unreachable" % c.KAFKA_BOOTSTRAP)
    assert topics.main(["delete"]) == 1


# -- topics.py against the broker --------------------------------------------------------------

@pytest.fixture
def scratch_topics():
    if not topics.broker_reachable():
        pytest.skip("Kafka at %s unreachable" % c.KAFKA_BOOTSTRAP)
    pytest.importorskip("kafka")
    tag = "u1test-%s." % uuid.uuid4().hex[:8]
    ts = (c.Topic(tag + "a", 1, 3600000, "test"), c.Topic(tag + "b", 2, 7200000, "test"))
    yield ts
    topics.delete_topics(c.KAFKA_BOOTSTRAP, ts)


def test_create_is_idempotent_and_counts_climb(scratch_topics):
    from kafka import KafkaProducer
    from kafka.admin import ConfigResource, ConfigResourceType

    a, b = scratch_topics
    assert topics.create_topics(c.KAFKA_BOOTSTRAP, scratch_topics) == {a.name: "created",
                                                                       b.name: "created"}
    assert topics.create_topics(c.KAFKA_BOOTSTRAP, scratch_topics) == {a.name: "exists",
                                                                       b.name: "exists"}

    def retention(admin, name):
        res = admin.describe_configs([ConfigResource(ConfigResourceType.TOPIC, name)])
        if isinstance(res, dict):  # kafka-python 3: {"topic": {name: {key: {"value": ...}}}}
            return res["topic"][name]["retention.ms"]["value"]
        return {e[0]: e[1] for e in res[0].resources[0][4]}["retention.ms"]  # 2.x

    admin = topics._admin(c.KAFKA_BOOTSTRAP)
    try:
        assert retention(admin, a.name) == str(a.retention_ms)
        # A topic that drifted from the contract (auto-created, edited) is brought back.
        admin.alter_configs([ConfigResource(ConfigResourceType.TOPIC, a.name,
                                            configs={"retention.ms": "1000"})])
        assert retention(admin, a.name) == "1000"
        topics.create_topics(c.KAFKA_BOOTSTRAP, scratch_topics)
        assert retention(admin, a.name) == str(a.retention_ms)
    finally:
        admin.close()

    counts = {r.name: r for r in topics.topic_counts(c.KAFKA_BOOTSTRAP, [a.name, b.name], 10)}
    assert (counts[a.name].partitions, counts[a.name].messages) == (1, 0)
    assert (counts[b.name].partitions, counts[b.name].messages) == (2, 0)

    producer = KafkaProducer(bootstrap_servers=c.KAFKA_BOOTSTRAP)
    for i in range(3):
        producer.send(a.name, key=b"job", value=b"%d" % i)
    producer.send(b.name, partition=1, value=b"x")
    producer.flush(10)
    producer.close()

    counts = {r.name: r for r in topics.topic_counts(c.KAFKA_BOOTSTRAP, [a.name, b.name, "u1test-none"])}
    assert counts[a.name].messages == counts[a.name].retained == 3
    assert counts[b.name].messages == 1
    assert counts["u1test-none"].partitions == -1

    assert set(topics.delete_topics(c.KAFKA_BOOTSTRAP, scratch_topics)) == {a.name, b.name}
    assert topics.delete_topics(c.KAFKA_BOOTSTRAP, scratch_topics) == []  # gone, not pending


# -- docker-compose.yml ------------------------------------------------------------------------

_SUBST = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _resolve(s):
    """Compose-style ${VAR:-default} substitution, from this process's environment."""
    return _SUBST.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), str(s))


@pytest.fixture(scope="module")
def compose():
    yaml = pytest.importorskip("yaml")
    with open(COMPOSE) as f:
        return yaml.safe_load(f)


def test_compose_services_match_layers(compose):
    want = {l.compose_service for l in c.LAYERS if l.compose_service}
    assert set(compose["services"]) == want


def test_compose_ports_match_contract(compose):
    host = {}
    for name, svc in compose["services"].items():
        for p in svc.get("ports", []):
            hp, _, cp = _resolve(p).rpartition(":")
            host[name] = (int(hp), int(cp))
    assert host == {"kafka": (c.KAFKA_EXTERNAL_PORT, 9094), "db": (c.DB_PORT, 5432),
                    "dashboard": (c.DASHBOARD_PORT, 8501)}
    used = {hp for hp, _ in host.values()} | {c.KAFKA_INTERNAL_PORT}
    assert not used & set(c.RESERVED_PORTS)
    assert not {cp for _, cp in host.values()} & set(c.RESERVED_PORTS)


def test_compose_kafka_listeners(compose):
    env = compose["services"]["kafka"]["environment"]
    listeners = env["KAFKA_LISTENERS"]
    advertised = _resolve(env["KAFKA_ADVERTISED_LISTENERS"])
    assert "INTERNAL://:%d" % c.KAFKA_INTERNAL_PORT in listeners
    assert "INTERNAL://kafka:%d" % c.KAFKA_INTERNAL_PORT in advertised
    assert advertised.endswith(":%d" % c.KAFKA_EXTERNAL_PORT)


def test_compose_python_services(compose):
    svcs = compose["services"]
    internal_dsn = "postgresql://hypersplat:hypersplat@db:5432/hypersplat"
    for name in ("kafka-init", "processor", "dashboard"):
        env = svcs[name]["environment"]
        assert env["HYPERSPLAT_KAFKA_INTERNAL"] == "kafka:%d" % c.KAFKA_INTERNAL_PORT
        assert env["HYPERSPLAT_DB_DSN_INTERNAL"] == internal_dsn
    cmd = {n: " ".join(svcs[n]["command"]) for n in ("kafka-init", "processor", "dashboard")}
    assert "hypersplat.streaming.topics" in cmd["kafka-init"] and "create" in cmd["kafka-init"]
    assert "-m hypersplat.streaming.processor" in cmd["processor"]
    assert "streamlit run hypersplat/streaming/dashboard.py" in cmd["dashboard"]
    assert svcs["processor"]["depends_on"]["kafka-init"]["condition"] == "service_completed_successfully"


# -- ops/up.sh ---------------------------------------------------------------------------------

def _dry_run(*args):
    env = dict(os.environ, HYPERSPLAT_PYTHON=sys.executable)
    out = subprocess.run(["bash", str(UP_SH), "--dry-run", *args], env=env, check=True,
                         capture_output=True, text=True, timeout=60).stdout
    return out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_up_sh_syntax():
    subprocess.run(["bash", "-n", str(UP_SH)], check=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("mode", ["local", "compose"])
def test_up_sh_dry_run_follows_start_order(mode):
    out = _dry_run("--mode", mode)
    assert "mode: %s (dry run)" % mode in out
    steps = re.findall(r"^\[(\d)/6\] ([a-z ]+):", out, re.M)
    assert [int(n) for n, _ in steps] == [1, 2, 3, 4, 5, 6]
    assert [s for _, s in steps] == STEPS
    blocks = dict(zip(STEPS, re.split(r"^\[\d/6\] .*$", out, flags=re.M)[1:]))
    if mode == "local":
        assert "infra.sh up" in blocks["broker"]
        assert "-m hypersplat.streaming.topics" in blocks["topics"]
        assert "-m hypersplat.streaming.processor" in blocks["processor"]
        assert "streamlit run" in blocks["dashboard"]
    else:
        assert "up -d kafka db" in blocks["broker"]
        assert "run --rm kafka-init" in blocks["topics"]
        assert "up -d --no-deps processor" in blocks["processor"]
        assert "up -d --no-deps dashboard" in blocks["dashboard"]
    assert ("hypersplat.streaming.train_models" in blocks["anomaly models"]
            or "skipping training" in blocks["anomaly models"])
    assert "hypersplat.streaming.replay" in blocks["producer"]
    assert "HYPERSPLAT_STREAMING=1" in blocks["producer"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_up_sh_down_keeps_broker_unless_all():
    out = _dry_run("down", "--mode", "compose")
    assert "stop dashboard processor" in out and " down" not in out.replace("mode:", "")
    assert re.search(r"docker-compose\.yml down$", _dry_run("down", "--all", "--mode", "compose"), re.M)
    assert "infra.sh" not in _dry_run("down", "--all", "--mode", "local")
