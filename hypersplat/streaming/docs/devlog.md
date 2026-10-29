# Streaming mini-project: dev log

Handoff notes for the next developer. Newest entry first. Source of truth for what is required:
the "Streaming Mini-Project: Policy & Process Map for HyperSplat" PDF (22 rows). Source of truth
for names, topics, ports, features and tables: `hypersplat/streaming/contract.py`.

## 2026-10-29: base + parallel build (in progress)

### Goal
Add a six-layer streaming stack beside the unchanged batch reconstruction:
data source -> Kafka -> processor -> IsolationForest -> PostgreSQL -> Streamlit.
Everything lives in `hypersplat/streaming/`. No file outside it is edited.

### Branches
- `feat/streaming` (base, commit `96cc8a3`): contract, producer, infra script, tests. Worker PRs target it.
- Final step: one PR `feat/streaming` -> `current`.
- The repo had unrelated dirty files (`demo/run_stream_demo.sh`, `examples/simple_trainer.py`,
  `examples/trainer_schedule.py`, `hypersplat/stream/vjepa_latent.py`). They were never staged.
- Worker worktrees sometimes started from an old commit or `main`. Each worker reset to `96cc8a3`.
  Check every PR's base before merging.

### Status of the 8 units
| # | Unit | State | PR |
|---|------|-------|----|
| 1 | Infra, topics, compose, runbook | running at time of writing | - |
| 2 | Live runtime hooks | running at time of writing | - |
| 3 | Recorder + replay producer | PR open, tested | #30 |
| 4 | Processor | running at time of writing | - |
| 5 | Anomaly models + contamination experiment | running at time of writing | - |
| 6 | Streamlit dashboard | PR open, tested | #29 |
| 7 | Health check + overhead bench | running at time of writing | - |
| 8 | Report + architecture docs | PR open, tested | #28 |

Update this table as PRs land.

### Key design decisions
1. **Runtime capture without editing existing code.** `autohook/sitecustomize.py` (enabled by
   `HYPERSPLAT_STREAMING=1` and PYTHONPATH) patches `StreamSession._on_event`
   (`hypersplat/stream/session.py:771`, called by the ring thread and the ACE pipe reader) and
   `SummaryWriter.add_scalar` (trainer subprocess). `session.py` and `simple_trainer.py` stay identical.
2. **Never block training.** `producer.Sender.send` is `put_nowait` on a bounded queue; one daemon
   thread talks to Kafka. Full queue or broker down: drop and count (`dropped` rides on later events).
3. **Only new poses are scored.** `CameraPose.source` is `pose|reloc|map|file|replay`;
   `SCORED_SOURCES` excludes `reloc`/`map`, because a refit moves every camera at once and would
   look like an anomaly.
4. **Old file hand-off stays** (`poses_final.txt`). `FILE_HANDOFF_ENABLED` defaults on and
   `STREAMING_ENABLED` defaults off. Remove the fallback only after one clean full Kafka run.
5. **Train/serve parity.** Feature functions live in the contract (`pose_features`,
   `training_features`) and are used by both `train_models.py` and `processor.py`.
6. **Idempotent processing.** Offsets commit after the DB commit; inserts are
   `ON CONFLICT (job_id, seq) DO NOTHING`, so a restart replays the tail without duplicates.

### Ports (none of the repo's 8080 / 8081 / 8092)
Kafka external 9094 (internal 29092), PostgreSQL 5433, Streamlit 8501.

### Environment on the dev machine (Fedora, no GPU use)
- Python env: `.venv_3dgrut` (py3.12); streaming deps in `hypersplat/streaming/requirements.txt`.
  The ACE env (`../.conda_envs/ace0`) is py3.8, so `contract.py`, `producer.py` and hooks must stay py3.8-syntax.
- Kafka + PostgreSQL via `bash hypersplat/streaming/ops/infra.sh up|status|down`
  (podman containers `hs-kafka`, `hs-db`; there is no compose plugin on this box).
  Per-worker databases `hypersplat_u1..u8` and topic prefix `u<N>.` were for parallel testing only.
- The PDF assumes Windows Docker Desktop plus WSL for ACE. Not tested here. On WSL without mirrored
  networking set `HYPERSPLAT_KAFKA` to the Windows host IP.

### Gotchas found
- **Live runs need `tb_every=10`.** The trainer defaults to 100, so snapshots would arrive every 100
  steps, not 10 (PDF row 3). `live.sh` passes `--trainer-arg tb_every=10`.
- **Replay relabels `source` as `replay`** by default. record1 is mostly `reloc` rows, which are not
  scored. Trade-off: a `map` burst could then score as a false alarm. `--keep-source` keeps the originals.
- **Replaying a named `--job-id` twice without `--reset`** restarts `seq` at 0 and the processor
  drops those events as duplicates. The default id is unique.
- **Dashboard needs Streamlit >= 1.42** (`st.metric(border=True)`); `requirements.txt` still says 1.40.
- **Dashboard liveness** assumes the processor updates `jobs.updated_at` on every write.
- `MAX_MESSAGE_BYTES` comment in `contract.py` says ~400 B per pose; real size is ~314 B.
- `contract.LAYERS` lists only `replay` as the data-source module (not the hooks).
- `hypersplat.flagged-problems` has no consumer in the project (dashboard reads PostgreSQL); it is an output topic.
- Never commit generated data: `models/`, `recordings/`, `results/`, `*.joblib` are git-ignored here.

### Remaining work (in order)
1. Merge all unit PRs into `feat/streaming`; fix the small contract notes above.
2. Run the full replay demo: `ops/up.sh` (broker, topics, models, processor, dashboard), then
   `replay.py --inject shuffle`. Expect `health` PASS, red markers on the dashboard, flow stops when
   the replay is paused, and no loss across a processor restart.
3. One short live GPU run, 2k steps, `live.sh --video`, under the GPU lock
   (`flock experiments/.gpu.lock`). Confirm the hooks fire and steps/s is unchanged (`bench_overhead`).
4. Fill the `TODO(measure)` placeholders in `docs/report.md` and tick `docs/checklist.md`.
5. Add the per-concern diagrams (`docs/diagrams/`) and short ADRs (`docs/adr/`): hook-by-patching,
   drop-not-block queue, score only new poses, keep the file fallback.
6. Open the PR `feat/streaming` -> `current`.

### Where to look
- Contract: `hypersplat/streaming/contract.py`; sender: `producer.py`.
- Plan and rationale: `docs/architecture.md`, `docs/report.md`, `docs/checklist.md` (PR #28).
- Approved plan: `~/.claude/plans/mighty-coalescing-dove.md` (local to the original dev machine).

### Update (usage limit reached mid-build)
PRs open into `feat/streaming`: #28 docs (U8), #29 dashboard (U6), #30 replay (U3), #31 infra/topics (U1),
#32 models (U5), #33 health/bench (U7). **Not yet reported: U2 hooks, U4 processor** (workers may still be running;
check `gh pr list`). Nothing is merged yet and the end-to-end replay demo has not been run.

Open issues to fix after merge:
- ~~Pose features depend on scene scale~~ Fixed: the model now sees `step_ratio` (step / median recent step), `turn_deg`,
  `conf_ratio`, `focal_ratio`; `step_dist` and `confidence` stay in the database. Replay of record1 with 6 shuffled poses:
  flagged 33/593 (was 307/593), injected caught 4/6 (was 0/6). Same-scene holdout at contamination 0.05: blur 40/40,
  shuffle 0/16 (was 16/16); at 0.10: all 56 caught, 0/36 false alarms. Default contamination left at 0.05; shuffles need 0.10.
- Training model misses rising loss (0/128); live logging every 10 steps vs 100 in past TB runs changes window scale.
- `producer.py`: kafka-python ignores `max_block_ms` while bootstrapping (sender thread blocks ~30 s on a dead broker);
  add `bootstrap_timeout_ms` to `PRODUCER_CONFIG`.
- Raise streamlit floor in requirements.txt to >=1.42.
- bench_overhead's +56% on/off number is meaningless (no real streaming-on run exists yet); need a real pair of runs.
- Compose stack never brought up (no compose CLI here); test on Windows Docker Desktop.
- `u7.hypersplat.camera-poses` on the local Kafka holds ~10.8k bench events; delete the `u*.` topics before the demo.
