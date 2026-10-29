# Submission checklist

This file maps each of the 22 rows of the "Policy & Process Map" PDF to the module that
implements it and the evidence to submit. All modules are under `hypersplat/streaming/`. Tick a
box once its evidence has been collected.

Evidence kinds:
- **screenshot**: a dashboard or terminal capture
- **recording**: part of the demo video
- **table**: a filled table in [report.md](report.md)
- **section**: a section of report.md or [architecture.md](architecture.md)
- **test**: a passing pytest

| ☐ | # | Area: what to engineer | Implemented in | Evidence |
|---|---|---|---|---|
| [ ] | 1 | Architecture: all six layers exist and connect | `contract.py` (`LAYERS`), `docker-compose.yml`, `ops/up.sh` | architecture.md §1 diagram; screenshot of all services running |
| [ ] | 2 | Data source: two events (pose, training snapshot) | `contract.py` (`CameraPose`, `TrainingSnapshot`), `hooks.py`, `autohook/sitecustomize.py`, `sources.py` | report.md §3 samples; architecture.md §3, §8; `tests/test_contract.py` |
| [ ] | 3 | Pace: training logged every 10 steps, replay at ~5 events/s | `contract.py` (`TRAIN_LOG_EVERY`, `REPLAY_RATE_HZ`), `hooks.py`, `replay.py`, `live.sh` (`--tb_every 10`) | screenshot of `training_history.step` advancing by 10; replay log showing ~5 events/s |
| [ ] | 4 | Guard bad values: drop and count; dropped counter on the dashboard | `contract.py` (`validate`), `producer.py`, `processor.py`, `dashboard.py` | `tests/test_producer.py`; screenshot of the dropped counter rising after a bad-value injection |
| [ ] | 5 | Kafka is mandatory | `docker-compose.yml` (`kafka`, KRaft), `producer.py` | screenshot of the broker running; architecture.md §2 |
| [ ] | 6 | Three named topics | `topics.py`, `contract.py` (`TOPICS`) | terminal capture of `kafka-topics --describe`; architecture.md §2 table |
| [ ] | 7 | Producer and consumer independent; the processor restarts and catches up | `processor.py` (`CONSUMER_CONFIG`, commit after DB write, idempotent inserts) | recording of a processor kill and restart; report.md §6.5 table |
| [ ] | 8 | Sliding window of 30 | `contract.py` (`WINDOW`), `processor.py` | report.md §5.1 |
| [ ] | 9 | Live metrics | `contract.py` (`POSE_METRICS`, `TRAINING_METRICS`), `processor.py`, `dashboard.py` | screenshot of metric panels; report.md §5.2 |
| [ ] | 10 | IsolationForest on poses and on training windows | `train_models.py`, `processor.py`, `contract.py` (`POSE_FEATURES`, `TRAINING_FEATURES`) | report.md §5.3; report.md §6.1 table |
| [ ] | 11 | Train offline, load at startup, never retrain in the stream | `train_models.py` (fit and save), `processor.py` (load only) | report.md §5.3; processor startup log showing the model paths loaded |
| [ ] | 12 | Contamination tuning with injected bad poses (shuffled frames, blurred stretch), 3 values | `experiment.py`, `replay.py`, `contract.py` (`CONTAMINATION_GRID`, `INJECTIONS`) | report.md §5.4 and §6.2 table |
| [ ] | 13 | PostgreSQL between processor and dashboard | `docker-compose.yml` (`db`), `processor.py` (writes), `dashboard.py` (reads) | architecture.md §4; screenshot of `psql` row counts |
| [ ] | 14 | History tables + one-row current snapshot per job | `contract.py` (`DDL`: `pose_history`, `training_history`, `jobs`) | architecture.md §4; screenshot of `SELECT * FROM jobs` |
| [ ] | 15 | Plan for growing tables | `contract.py` (`DDL` indexes, `archived`, `DASHBOARD_ROWS`) | architecture.md §4 "Growth plan" |
| [ ] | 16 | Streamlit is mandatory | `dashboard.py` | screenshot of the dashboard at `localhost:8501` |
| [ ] | 17 | Auto-refresh every 2 s with a fresh connection | `dashboard.py` (`DASHBOARD_REFRESH_S`) | recording of the dashboard updating while the producer runs |
| [ ] | 18 | Red markers on the loss curve; top-down camera path with bad poses red | `dashboard.py` (`ANOMALY_COLOR`, `camera_center`) | screenshot of both charts with red markers after injection |
| [ ] | 19 | Fixed start order: broker, topics, model training, processor, dashboard, producer | `ops/up.sh`, `contract.py` (`LAYERS`), `docs/runbook.md` | architecture.md §5; terminal capture of `ops/up.sh` |
| [ ] | 20 | One-command health check | `health.py` (`HEALTH_QUERY`) | report.md §6.4 output |
| [ ] | 21 | Demo: reset, live updates, pause the producer, inject bad poses | `ops/up.sh`, `replay.py`, `live.sh`, `docs/runbook.md` | demo recording |
| [ ] | 22 | Deliverables: code, report, demo | everything under `hypersplat/streaming/`, `docs/report.md` | PR merged; report.md complete with no `TODO(measure)` left; demo recording |

Before submitting, also check:

- [ ] `pytest hypersplat/streaming/tests -q` passes.
- [ ] Throughput with streaming on and off is measured (`bench_overhead.py`, report.md §6.3).
- [ ] No generated data is committed: models, recordings and results are covered by
      `hypersplat/streaming/.gitignore`.
- [ ] The file hand-off fallback is still on (`HYPERSPLAT_FILE_HANDOFF=1`) for the demo run.
