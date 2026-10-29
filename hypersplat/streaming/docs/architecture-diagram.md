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

    subgraph K["Apache Kafka broker (KRaft, one node)"]
        direction LR
        A[["hypersplat.camera-poses<br/>key = job_id, retention 7 d"]]
        B[["hypersplat.training-measurements<br/>key = job_id, retention 7 d"]]
        C[["hypersplat.flagged-problems<br/>key = job_id, retention 30 d"]]
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
    S1 -- "Anomaly (reason + score)" --> C
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
        WIN["window: last 30 events per job_id<br/>(applyInPandasWithState keeps the sliding window)"]
        UDF["pandas UDF scores each row with the same<br/>joblib models + rising-loss rule<br/>(models read from a mounted volume)"]
        FB["foreachBatch<br/>upsert to Postgres with ON CONFLICT (job_id, seq)<br/>and publish flagged events to Kafka"]
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
| Exactly-once | offset commit after the database write, plus `ON CONFLICT DO NOTHING` | checkpoint plus the same idempotent upsert |
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
    DBc -. "StreamedResults<br/>(flags for this run's job)" .-> DEMO
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
