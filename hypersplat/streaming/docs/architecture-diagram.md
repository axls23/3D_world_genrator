# Streaming architecture: the six layers on Apache Kafka (pub/sub) and Spark

The course guide (Real-Time Streaming Analytics) asks for six layers: data source, streaming tool,
stream processor, ML model, database, live dashboard. This is how HyperSplat fills each one.
Green boxes are built and verified; the dashed Spark box is the alternative processor design and is
**not built yet** (the processor that runs today is plain Python, which the guide allows).

## 1. The six layers at a glance

```mermaid
flowchart LR
    subgraph L1["Layer 1 · Data source"]
        direction TB
        ACE["ACE pose estimator<br/>one CameraPose per registered frame"]
        TR["3DGS trainer<br/>one TrainingSnapshot every 10 steps"]
        RP["Replay of a recorded run<br/>5 events per second (no GPU)"]
    end

    subgraph L2["Layer 2 · Streaming tool · Apache Kafka"]
        direction TB
        T1[["topic<br/>hypersplat.camera-poses"]]
        T2[["topic<br/>hypersplat.training-measurements"]]
        T3[["topic<br/>hypersplat.flagged-problems"]]
    end

    subgraph L3["Layer 3 · Stream processor"]
        direction TB
        W["Sliding window of 30 events<br/>rolling loss, loss trend, step ratio,<br/>running peak, confidence"]
    end

    subgraph L4["Layer 4 · ML model"]
        direction TB
        IF["Isolation Forest (pose)<br/>Isolation Forest (training)<br/>trained offline, loaded once"]
        RL["Rising-loss trend rule<br/>Mann-Kendall over 8 snapshots"]
    end

    subgraph L5["Layer 5 · Database · PostgreSQL"]
        direction TB
        J[("jobs<br/>one current row per run")]
        H[("pose_history<br/>training_history<br/>every event + anomaly flag")]
    end

    subgraph L6["Layer 6 · Live dashboards"]
        direction TB
        ST["Streamlit :8501<br/>KPI cards, loss curve,<br/>red markers, 2 s refresh"]
        V["8090 Streaming visualization<br/>pose map, camera feed,<br/>flags for this run"]
    end

    ACE -- publish --> T1
    TR -- publish --> T2
    RP -- publish --> T1
    RP -- publish --> T2
    T1 -- subscribe --> W
    T2 -- subscribe --> W
    W --> IF
    W --> RL
    IF --> H
    RL --> H
    W --> J
    IF -- publish flagged --> T3
    RL -- publish flagged --> T3
    H -- fresh connection per refresh --> ST
    J --> ST
    H -- StreamedResults --> V

    classDef built fill:#e6f7ef,stroke:#2f9a52,color:#12361f;
    class ACE,TR,RP,T1,T2,T3,W,IF,RL,J,H,ST,V built;
```

## 2. Kafka publish / subscribe in detail

Publishers never talk to the processor and the processor never talks to the dashboards. Each side only
knows Kafka (or Postgres), so any part can restart or run at a different speed without losing events.

```mermaid
flowchart TB
    subgraph PUB["Publishers (producer.Sender: queue, background thread, drops instead of blocking)"]
        P1["ACE sidecar hook<br/>hooks.py wraps StreamSession._on_event"]
        P2["Trainer hook<br/>hooks.py wraps SummaryWriter.add_scalar('train/*')"]
        P3["replay.py<br/>plays a recording at a fixed rate"]
    end
    LOSS["Loss boundary 1: the sender queue is full or Kafka is down<br/>the event is dropped and counted, never blocks the trainer<br/>Sender.stats(): sent, dropped, queue depth, broker up"]
    PUB --- LOSS

    subgraph K["Apache Kafka broker (KRaft, one node)"]
        direction LR
        A[["hypersplat.camera-poses<br/>key = job_id, retention 7 d"]]
        B[["hypersplat.training-measurements<br/>key = job_id, retention 7 d"]]
        C[["hypersplat.flagged-problems<br/>key = job_id, retention 30 d"]]
        D[["hypersplat.dead-letter<br/>bad messages with reason + first 256 chars"]]
    end

    subgraph SUB["Subscribers"]
        direction TB
        S1["processor<br/>consumer group hypersplat-processor<br/>commits an offset only after the database write"]
        S2["any extra reader<br/>(own consumer group,<br/>for example a notifier)"]
    end

    P1 -- "CameraPose" --> A
    P3 -- "CameraPose" --> A
    P2 -- "TrainingSnapshot" --> B
    P3 -- "TrainingSnapshot" --> B
    A --> S1
    B --> S1
    S1 -- "Anomaly (reason, score, anomaly_id)" --> C
    S1 -- "DeadLetter (undecodable or out-of-range)" --> D
    C -.-> S2

    N1["Why this helps<br/>• a paused producer freezes the dashboard numbers (liveness demo)<br/>• restarting the processor resumes from the last committed offset<br/>• (job_id, seq) is unique in Postgres, so a replayed tail adds no duplicates"]
    S1 --- N1
```

## 3. The same pipeline with Spark as the stream processor (design, not built)

The guide lists Spark and Flink as alternatives for layer 3. Only the middle changes: Kafka stays the
transport, Postgres stays the serving layer, and the models and dashboards are untouched.

```mermaid
flowchart LR
    K1[["Kafka<br/>camera-poses"]]
    K2[["Kafka<br/>training-measurements"]]

    subgraph SP["Spark Structured Streaming job (replaces the Python processor container)"]
        direction TB
        R["readStream (kafka source)<br/>consumer offsets in a checkpoint dir"]
        PA["parse JSON with the contract schema,<br/>drop bad rows (the dropped-event counter)"]
        WIN["state key = (job_id, event_type): pose and training<br/>windows kept apart; the 30-event deque, eviction and<br/>restore are our code inside applyInPandasWithState"]
        UDF["pandas UDF scores each row with the same<br/>joblib models + rising-loss rule<br/>(model version stamped on every result)"]
        FB["foreachBatch (replay-safe, not atomic)<br/>Postgres upsert ON CONFLICT, then Kafka publish<br/>flagged events carry a stable anomaly_id"]
        R --> PA --> WIN --> UDF --> FB
    end

    K1 --> R
    K2 --> R
    FB --> PG[("PostgreSQL<br/>same tables")]
    FB --> K3[["Kafka<br/>flagged-problems"]]
    PG --> D["Streamlit + 8090 view<br/>(unchanged)"]

    classDef planned fill:#fff,stroke:#8999ad,stroke-dasharray: 5 5,color:#334;
    class R,PA,WIN,UDF,FB planned;
```

| | Python processor (built) | Spark Structured Streaming (planned) |
|---|---|---|
| Fits | one machine, one job, 5 to 100 events per second | many jobs or higher rates, needs a cluster |
| State | in-memory deques, rebuilt from Postgres on restart | managed state plus a checkpoint |
| Delivery | at-least-once: offset commit after the database write, replay-safe idempotent inserts | at-least-once: checkpoint plus the same idempotent upsert; the two sinks are not one transaction |
| Cost | one small container | a JVM and Spark, much heavier for a course demo |

## 4. Where it runs

```mermaid
flowchart TB
    subgraph HOST["Host (GPU, camera, WSL / Windows)"]
        DEMO["run_stream_demo.sh --streaming<br/>pipeline + trainer + ACE<br/>8090 base page and Streaming visualization view"]
    end

    subgraph POD["podman pod hypersplat-streaming (ops/pod.sh up)"]
        direction TB
        KF["kafka :9094"]
        DBc["postgres :5433"]
        PR["processor<br/>models from the hypersplat-models volume"]
        SL["streamlit :8501"]
    end

    DEMO -- "events to localhost:9094" --> KF
    KF --> PR
    PR --> DBc
    DBc --> SL
    DBc -. "SQL read by StreamedResults inside the host process<br/>(poll every 1.5 s, not a push channel)" .-> DEMO
```

## Reading guide

| Layer | Guide's requirement | Built as | Code |
|---|---|---|---|
| 1 Data source | any event stream, paced | hooks on the live run, or a recorded replay | `hooks.py`, `sources.py`, `replay.py` |
| 2 Streaming tool | Kafka, Storm, Spark or Flink | Apache Kafka, 3 named topics | `topics.py`, `producer.py`, `contract.py` |
| 3 Stream processor | live metrics, sliding window | Python, 30-event windows | `processor.py` |
| 4 ML model | at least one algorithm | 2 Isolation Forests plus a rising-loss rule | `train_models.py`, `contract.rising_loss` |
| 5 Database | processor writes, dashboard reads | PostgreSQL, history plus one-row snapshot | `contract.DDL` |
| 6 Dashboard | auto-refresh, red markers | Streamlit (8501) and the 8090 view | `dashboard.py`, `hypersplat/stream/dashboard.py` |

## 5. The processor's internal graph (what each edge guarantees)

Pose and training events are processed independently. They are two keyed streams, never one merged
window, so a burst of poses cannot push training snapshots out of their window.

```mermaid
flowchart TB
    IN1[["camera-poses<br/>key job_id · 1 partition"]]
    IN2[["training-measurements<br/>key job_id · 1 partition"]]

    DEC["decode + validate<br/>schema version, finite numbers, DB range"]
    DLQ[["dead-letter topic"]]

    subgraph PS["State key (job_id, pose)"]
        direction TB
        PW["window: last 30 poses by arrival order<br/>step_ratio, turn_deg, conf_ratio, focal_ratio"]
        PM["pose Isolation Forest"]
        PW --> PM
    end
    subgraph TS["State key (job_id, training)"]
        direction TB
        TW["window: last 30 snapshots by arrival order<br/>loss trend, growth, memory"]
        TM["training Isolation Forest<br/>+ rising-loss rule"]
        TW --> TM
    end

    DB[("PostgreSQL transaction per poll batch<br/>INSERT ... ON CONFLICT (job_id, seq) DO NOTHING<br/>jobs row: counters only for rows actually inserted")]
    OFF["commit Kafka offsets"]
    OUT[["flagged-problems<br/>anomaly_id = hash(job, source, seq, model)"]]

    IN1 --> DEC
    IN2 --> DEC
    DEC -- "bad" --> DLQ
    DEC -- "pose" --> PW
    DEC -- "training" --> TW
    PM --> DB
    TM --> DB
    DB -- "1 commit" --> OUT
    OUT -- "2 flush" --> OFF
```

| Edge | Schema / key | State | Delivery | If it fails |
|---|---|---|---|---|
| producer to topic | `CameraPose` / `TrainingSnapshot`, key `job_id`, per-type `seq` | none | at-most-once at the producer queue | event dropped and counted in `Sender.stats()`; the dashboard shows the producer drop counter |
| topic to decode | JSON, `v` schema version | none | at-least-once (offsets commit last) | bad message goes to the dead-letter topic with its reason, the batch continues |
| decode to window | typed event | count-based window per (job, type), ordered by arrival and `seq` | n/a | `seq` at or below the last handled is skipped and counted (`Processor.skipped`), never silent |
| window to Postgres | history row plus anomaly flag | rebuilt from the newest 30 history rows on restart | idempotent: unique `(job_id, seq)` per table, counters add only for inserted rows | transaction rolls back, windows are cleared and rebuilt, offsets stay uncommitted, the batch replays |
| Postgres to flagged topic | `Anomaly` with `anomaly_id` | none | at-least-once: sent after the DB commit, flushed before offsets commit | replay re-sends the same `anomaly_id`; consumers dedupe on it |
| flagged topic to offsets | n/a | n/a | n/a | if an anomaly is not delivered, offsets are not committed and the batch is retried |

## 6. Delivery contract, stated exactly

```mermaid
sequenceDiagram
    participant K as Kafka (input topics)
    participant P as Processor
    participant D as PostgreSQL
    participant A as flagged-problems topic

    K->>P: poll batch
    P->>D: BEGIN, idempotent inserts, job counters
    P->>D: COMMIT
    Note over P,D: crash here: batch replays, inserts do nothing, nothing double counted
    P->>A: send anomalies (anomaly_id), flush
    Note over P,A: crash here or send fails: batch replays, same anomaly_ids sent again
    P->>K: commit offsets
    Note over P,K: crash here: batch replays (same as above)
```

- **What is guaranteed:** every event that reaches Kafka is processed at least once; the history tables
  and job counters end up the same however often a batch replays; a flagged event is always re-sent
  until delivered.
- **What is not guaranteed:** exactly-once across Kafka, PostgreSQL and the output topic. They are
  three systems with no shared transaction, so the flagged-problems topic can contain duplicates. That
  is why every anomaly has a stable `anomaly_id`, and why Postgres is the authoritative store: the
  dashboards read Postgres, not the topic.
- **Where events can still be lost:** before Kafka, when the producer queue is full or the broker is
  down (a deliberate choice so training is never slowed; counted, not hidden).
- **Window semantics:** count-based (the last 30 events of one type for one job), ordered by arrival.
  The loss-trend rule needs 8 consecutive training snapshots, never a mix. Late or duplicate events
  are skipped and counted; there is no event-time watermark because one producer per job on one
  partition keeps order. Switching to event-time windows is the first change a Flink port would make.

## 7. Topology and how it would scale

| Topic | Partitions now | Key | Consumer group | Limit |
|---|---|---|---|---|
| camera-poses | 1 | `job_id` | `hypersplat-processor` | one processor instance reads it; ordering per job is exact |
| training-measurements | 1 | `job_id` | `hypersplat-processor` | same |
| flagged-problems | 1 | `job_id` | none yet | readers are optional |
| dead-letter | 1 | `job_id` | none | inspection only |

To scale, raise partitions to the number of processor replicas and keep `job_id` as the key: jobs
spread across replicas, but one hot job still sits on one partition. Replication factor is 1 and there is
one broker, so a broker failure loses availability and data; this is a demo topology, not an HA one.
The partition-change step needs the processor's rebalance handler (it already clears per-job state on
rebalance and rebuilds from Postgres).

## 8. What the 8090 view reads

The Streaming visualization view does not receive a pushed stream. `StreamedResults` runs a short SQL
query against PostgreSQL (cached 1.5 s, pinned to `HYPERSPLAT_JOB_ID`), so Postgres owns the truth, a
restart of the visualizer loses nothing, and a database outage shows as "waiting for processor results"
with the reason instead of zero flags.

## 9. Operational signals

| Signal | Where |
|---|---|
| producer sent, dropped, queue depth, broker up | `Sender.stats()` |
| consumer lag, events/s, drops, skipped, output queue | processor log line every 10 s (`consumer_lag`) |
| undecodable or out-of-range messages | dead-letter topic and `jobs.events_dropped` |
| liveness and counts | `health.py` and the dashboard header |

## 10. Next steps from the review (not done)

1. A forced-failure test pass against real Kafka and Postgres: kill the processor between each step of
   the sequence in section 6 and compare the three stores.
2. Make the jobs snapshot update conditional on a newer `seq`, so a replay of an older batch can never
   move `current_step` backwards (today the in-process guard and the history rebuild prevent it).
3. Add `event_id` and a model version to the output event (the version is only in the model file today).
4. Benchmark the Python processor before porting to Spark or Flink; port only for a measured reason.
