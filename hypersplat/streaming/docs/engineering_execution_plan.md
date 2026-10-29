# Engineering execution plan for the missing PDF streaming architecture

## 1. Current system map

The repository already contains a working live demo path, but it is not the same as the PDF’s six-layer system design.

Current runtime components:

- Live 8090 shell dashboard: `hypersplat/stream/dashboard.py`
  - Serves the HTML shell, `/status.json`, `/metrics.json`, frame endpoints, and the embedded viewer iframe.
  - Loads the viewer via JavaScript from the configured viewer port.
- Live viewer process: separate from the dashboard shell, usually bound to a viewer port such as 8080.
- Training/session runtime: `examples/simple_trainer.py` and the broader `hypersplat/stream` runtime
  - Provides the session state, live counters, and telemetry used by the dashboard.
- Streaming mini-project contract and producer layer: `hypersplat/streaming/contract.py`, `hypersplat/streaming/producer.py`
  - Defines the shared event schema, Kafka topics, and bounded async producer queue.
- Thin Streamlit wrapper: `hypersplat/streaming/streamlit_dashboard.py`
  - Reads `/status.json` from the 8090 dashboard and renders a simple wrapper UI.

This means the repo has:

- a live custom dashboard path
- a partial event schema and Kafka-ready contract
- a Streamlit wrapper

But it does not yet have the end-to-end PDF architecture:

- Kafka broker + topics + runtime producers
- processor pipeline that scores pose/training events
- anomaly model training and scoring
- PostgreSQL persistence and history queries
- full Streamlit dashboard that reads from the database and refreshes live

## 2. Required architecture from the PDF

The PDF’s design is a six-layer flow:

1. Data source
2. Kafka
3. Processor
4. Anomaly model
5. Database
6. Streamlit dashboard

The intended flow is:

`data source -> Kafka topic(s) -> processor -> model.score -> database -> dashboard`

The project’s actual code already reflects that intent in `hypersplat/streaming/contract.py`, but the runtime glue is still missing.

## 3. Engineering work required to close the gap

### Phase A — Kafka data plumbing

Scope:
- Create or verify the Kafka broker setup for the project environment.
- Ensure the producer sends pose events and training snapshots under the defined topics.
- Keep the old file handoff as a fallback until the streaming path is proven stable.

Files to target:
- `hypersplat/streaming/contract.py`
- `hypersplat/streaming/producer.py`
- `hypersplat/streaming/ops/infra.sh`

Acceptance criteria:
- A producer can emit events to `TOPIC_POSES`, `TOPIC_TRAINING`, and `TOPIC_ANOMALIES` without blocking the trainer.
- Event drops are counted and visible as `dropped` metrics.
- A broker outage does not stall the training loop.

Dependencies:
- None beyond the contract and producer design already present.

### Phase B — Processor and anomaly model pipeline

Scope:
- Build a processor service that consumes Kafka events from the pose and training topics.
- Extract the feature vectors defined in the contract (`pose_features`, `training_features`).
- Run the isolation forest / anomaly model against the rolling window data.
- Emit anomaly records to the database and/or anomaly topic.

Files to target:
- `hypersplat/streaming/contract.py` (feature contracts)
- new processor module under `hypersplat/streaming/`
- new model training module under `hypersplat/streaming/`

Acceptance criteria:
- Pose events are filtered to scored sources only.
- Model scores are generated on a rolling-window basis.
- Anomalies include reason text and source metadata.
- Model training is reproducible and can be retrained with injected anomalies for validation.

Dependencies:
- Phase A must be stable enough to feed the processor.

### Phase C — PostgreSQL persistence layer

Scope:
- Create the PostgreSQL schema for jobs, pose history, training snapshots, and anomaly flags.
- Add idempotent inserts keyed by `(job_id, seq)` to avoid duplicate replay issues.
- Provide the queries needed for the dashboard health and trend panels.

Files to target:
- new database model/schema under `hypersplat/streaming/`
- dashboard query layer

Acceptance criteria:
- Processed events persist in a reliable schema.
- Restarting the processor does not create duplicate rows.
- The dashboard can read a live summary and recent anomaly history from PostgreSQL.

Dependencies:
- Phase B outputs stored model decisions.

### Phase D — Streamlit dashboard implementation

Scope:
- Replace the wrapper-like dashboard with a true PDF-aligned dashboard backed by PostgreSQL and live data refresh.
- Add panels for:
  - job health
  - rolling training metrics
  - anomaly feed
  - flagged sources
  - recent event counts and drop counters
- Provide refresh/auto-update and live status cards.

Files to target:
- `hypersplat/streaming/streamlit_dashboard.py`
- new dashboard data access helpers

Acceptance criteria:
- Dashboard reads live state from the database rather than only the 8090 demo endpoint.
- Refreshes every few seconds without manual restart.
- Displays anomaly state and health indicators clearly.

Dependencies:
- Phase C must complete and provide queryable data.

### Phase E — validation and replay testing

Scope:
- Run a replay test with injected anomalies.
- Validate that the pipeline survives processor restart and continues without duplication.
- Confirm the dashboard shows red markers and health changes as expected.

Acceptance criteria:
- Replay with injected anomalies produces known flagged events.
- Health checks pass.
- No duplicate processing occurs.

Dependencies:
- All prior phases.

## 4. Recommended execution order

1. Finish the Kafka contract and producer hardening.
2. Build the processor plus feature extraction.
3. Add the PostgreSQL schema and idempotent write path.
4. Build the dashboard around persisted data.
5. Run replay and live validation against the final end-to-end data flow.

## 5. Immediate action items from the current repo state

- Keep the current custom 8090 dashboard working as a smoke-test path.
- Use the existing `hypersplat/streaming/contract.py` as the canonical model contract.
- Upgrade the dashboard dependency floor to Streamlit 1.42 in `hypersplat/streaming/requirements.txt`.
- Treat the current `streamlit_dashboard.py` as a transitional wrapper, not the final PDF-compliant dashboard.

## 6. Bottom line

The repo already contains a live telemetry demo and the foundation of a Kafka-aware event model. The missing engineering work is not a single bug fix; it is the implementation of the remaining end-to-end architecture from the PDF: Kafka ingestion, processor/model scoring, database persistence, and a true Streamlit dashboard fed from persisted job data.
