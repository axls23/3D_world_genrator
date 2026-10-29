# Streaming anomaly monitoring for HyperSplat reconstructions

Report for the "Policy & Process Map" streaming mini-project (PDF row 22). Wiring, ports and
tables are in [architecture.md](architecture.md). Evidence for each PDF row is listed in
[checklist.md](checklist.md). Values marked `TODO(measure)` have not been measured yet.

## 1. Abstract

HyperSplat turns a video into a 3D Gaussian Splatting scene in batch: camera poses from ACE-Zero,
then depth, then 3DGS training, then a viewer. A bad pose or a diverging training run currently
shows up only when someone opens the finished scene. We added a six-layer streaming path beside
the pipeline: hooks in the pose estimator and trainer, Apache Kafka, a Python processor with
sliding windows of 30 events, two Isolation Forest models, PostgreSQL and a Streamlit
dashboard. Only small numeric events are streamed. Producers never block the trainer and drop
events they cannot deliver. The models are fitted offline and only scored online. We tune
contamination with injected bad poses (shuffled frames, blurred stretches). Detection quality,
throughput overhead and restart behaviour are reported in section 6.

## 2. Introduction

Two failure modes cost the most time in this pipeline:

1. **Bad camera poses.** ACE sometimes places a frame far from its neighbours, rotates it 180°
   or gives it low confidence. Training then fits floaters to that view.
2. **Training trouble.** Loss stalls or climbs, the Gaussian count grows without bound, or GPU
   memory creeps up until the run fails.

Both can be detected from numbers the pipeline already produces: the pose line ACE writes for
every frame, and the scalars the trainer logs to TensorBoard. The project streams those numbers,
scores them as they arrive, and shows the result within seconds. It must not slow training or
change the reconstruction. The constraints from the brief are: Kafka and Streamlit are
mandatory, PostgreSQL sits between processor and dashboard, the models are never retrained in
the stream, and the existing file hand-off stays as a fallback.

## 3. Dataset

The dataset is the stream of two event types (PDF row 2), defined in
`hypersplat/streaming/contract.py`. Both are keyed by `job_id` and numbered by `seq`, so a gap
in `seq` means lost events.

**CameraPose**, one per registered frame, on `hypersplat.camera-poses`. Sources:
- live ACE sidecar events (`source = pose | reloc | map`)
- `poses_final.txt` lines (`file`)
- recordings (`replay`)

Fields: frame name, world-to-camera quaternion and translation, focal length in px, ACE
confidence (inlier count), producer drop counter, and the ground-truth injection label. This
sample was made from a real ACE line,
`/data/drone/frames/000184.jpg 0.9962 0.0123 -0.0851 0.0087 0.4127 -0.0318 1.2645 1142.7 2316`:

```json
{"job_id":"drone-2026-10-29","seq":183,"ts":1793260812.41,"frame":"000184.jpg","qw":0.9962580685938414,"qx":0.012300716968183346,"qy":-0.08510496048718721,"qz":0.008700507123837,"tx":0.4127,"ty":-0.0318,"tz":1.2645,"focal":1142.7,"confidence":2316.0,"dropped":0,"injected":null,"source":"pose","kind":"pose","v":1}
```

**TrainingSnapshot**, one every `contract.TRAIN_LOG_EVERY` = 10 trainer steps, on
`hypersplat.training-measurements`. Field names mirror the trainer's TensorBoard tags
(`contract.TB_TAGS`): loss, Gaussian count, peak GPU memory, L1, SSIM loss, gradient norm, plus
steps/s for the overhead measurement.

```json
{"job_id":"drone-2026-10-29","seq":420,"ts":1793260840.07,"step":4200,"loss":0.0612,"num_gaussians":412530,"gpu_mem_gb":5.83,"l1":0.0418,"ssim":0.1387,"grad_norm":null,"steps_per_s":21.4,"dropped":0,"injected":null,"kind":"training","v":1}
```

The processor emits a third type, **Anomaly**, on `hypersplat.flagged-problems`:

```json
{"job_id":"drone-2026-10-29","ts":1793260812.47,"source":"pose","source_seq":183,"model":"pose_iforest","score":-0.0731,"reason":"turn_deg=74.2","injected":"shuffle","kind":"anomaly","v":1}
```

**Training data for the models.** Past reconstructions under `results/`:
- their `poses_final.txt` files, which give CameraPose sequences
- their TensorBoard event files, read through `contract.TB_TAGS`, which give TrainingSnapshot
  sequences

The processor's feature functions turn both into windows of the same shape it sees live.
`recording.py` exports live runs so they can be replayed (`replay.py`, 5 events/s) for the demo
and the contamination experiment. Recordings and fitted models are generated data and are not
committed.

**Data quality (row 4).** `contract.validate` rejects:
- non-finite required fields
- negative counters
- a quaternion with norm outside (0.9, 1.1)
- a non-positive focal length, a negative confidence or a missing frame name
- negative loss
- unknown pose sources

Invalid optional training measurements are set to null and the event is kept. Rejected events
are dropped and counted. A frame that ACE could not localize is counted separately as
unregistered, because it is a normal outcome.

## 4. Architecture

Six layers, each its own process, connected only through the contract:

| Layer | Tool |
|---|---|
| Data source | ACE pose estimator hooks, 3DGS trainer hooks, replay producer |
| Streaming | Apache Kafka (KRaft), three topics keyed by `job_id` |
| Processor | Python consumer with sliding windows |
| Anomaly model | scikit-learn IsolationForest ×2 (poses, training windows) |
| Database | PostgreSQL: `pose_history`, `training_history`, one-row-per-job `jobs` |
| Dashboard | Streamlit, refreshes every 2 s |

The batch reconstruction is unchanged. The hooks are installed at runtime by a `sitecustomize`
module that patches `StreamSession._on_event` and `SummaryWriter.add_scalar`, so no file
outside `hypersplat/streaming/` was edited. Sends go into a bounded in-process queue and a
background thread drains it, so a slow or missing broker costs the trainer only a
`put_nowait`. Start order:
1. broker
2. topics
3. database
4. model training
5. processor
6. dashboard
7. producer

The diagram, topic table, DDL, growth plan, ports and the two-OS setup are in
[architecture.md](architecture.md).

## 5. Methodology

### 5.1 Sliding window

The processor keeps the last `contract.WINDOW` = 30 events per job and per event type. The
rationale, copied from the contract:

> 30 snapshots = 300 trainer steps at TRAIN_LOG_EVERY=10, long enough to see a loss trend
> through per-batch noise yet short enough to react within a minute; 30 poses ~ one second of
> 30 fps keyframes, about one camera motion segment.

Below `contract.MIN_WINDOW_TO_SCORE` = 5 events, only metrics are computed and nothing is
scored.

### 5.2 Live metrics (row 9)

These are computed per event and written to the history tables. The column names are
`contract.POSE_METRICS` and `contract.TRAINING_METRICS`.

| Metric | Definition |
|---|---|
| `step_dist` | Distance between this camera centre and the previous one, C = -Rᵀt |
| `turn_deg` | Rotation angle between consecutive poses, 2·acos(\|q₁·q₂\|) |
| `conf_roll` | Rolling mean ACE confidence over the window |
| `loss_roll` | Rolling mean loss over the window |
| `loss_trend` | Least-squares slope of loss over the window, divided by its mean |
| `gs_growth` | Gaussians added per trainer step across the window |
| `best_loss` | Lowest loss so far in the job |
| `peak_mem_gb` | Highest `gpu_mem_gb` so far in the job |

The `jobs` row also holds running counters: accepted and rejected poses, dropped events and
anomalies. The dashboard shows the dropped counter (row 4).

### 5.3 Isolation Forest models (rows 10, 11)

There are two models, one per event type. Each scores a feature vector built from the current
window:

- **Pose model** (`contract.POSE_MODEL` = `pose_iforest`), with features
  `contract.POSE_FEATURES`:
  - `step_dist` and `turn_deg`, which catch jumps and flips
  - `confidence` and `conf_ratio` (confidence divided by the window mean), which catch weak
    registrations
  - `focal_ratio` (focal divided by the window median), which catches a broken intrinsics
    estimate
- **Training model** (`contract.TRAINING_MODEL` = `training_iforest`), with features
  `contract.TRAINING_FEATURES`:
  - `loss_ratio_best`: current loss divided by the window minimum
  - `loss_trend`
  - `loss_cv`: coefficient of variation of loss
  - `gs_growth`
  - `mem_delta_gb`

The feature functions (`contract.pose_features`, `contract.training_features`) live in the
contract and are imported by both the offline trainer and the processor. That removes
train/serve skew by construction.

**Offline training and online serving are separate.** `train_models.py` fits
`IsolationForest(n_estimators=200, random_state=0, contamination=c)` (`contract.IFOREST_PARAMS`)
on windows from past runs. It writes `models/pose_iforest.joblib` and
`models/training_iforest.joblib` (`contract.model_path`). The processor loads both once at
startup and only calls `decision_function` and `predict`. It never calls `fit`. An event is
flagged when `predict` returns -1, which happens when the decision score is below 0. The flagged
event gets `is_anomaly = true` in its history row and an Anomaly message on
`hypersplat.flagged-problems`. To change a model, retrain offline and restart the processor.

Only `pose`, `file` and `replay` poses are scored (`contract.SCORED_SOURCES`). `reloc` and
`map` events are bursts in which a refit moves many cameras at once. They are stored and drawn
but not scored, because their jumps are expected.

### 5.4 Contamination tuning (row 12)

`contamination` sets the fraction of training windows the forest treats as outliers, which
places the decision threshold. The plan is carried out by `experiment.py`:

1. Pick held-out clean runs that were not used for fitting.
2. Make injected copies. Each CameraPose carries its label in `injected`, which gives ground
   truth. `contract.INJECTIONS` defines two kinds:
   - **shuffle**: swap frames out of order inside a segment. This produces large `step_dist`
     and `turn_deg`.
   - **blur**: blur a contiguous stretch of frames before ACE, then re-localize. This produces
     low `confidence` and `conf_ratio`, and jitter.
3. For each value in `contract.CONTAMINATION_GRID` = (0.01, 0.05, 0.10):
   - fit the pose model on the clean training set
   - replay the injected runs and score them
   - count true and false positives against `injected`
4. Report per value:
   - precision and recall per injection kind
   - false-positive rate on clean frames
   - flagged events per 1 000
5. Choose the value with the best F1 that keeps the clean false-positive rate acceptable for a
   dashboard that someone actually reads. The default is `contract.CONTAMINATION` = 0.05 until
   the measurement is in.

### 5.5 Throughput and robustness checks

- **Overhead (constraint "training speed comes first").** `bench_overhead.py` runs the same
  training job with streaming off and on (`HYPERSPLAT_STREAMING=0/1`) and compares steps/s.
- **Health check (row 20).** `health.py` runs `contract.HEALTH_QUERY` twice. It passes when the
  pose and snapshot counts climb between the two runs and the anomaly count is above 0.
- **Restart (row 7).** Kill the processor during a replay, keep producing, then restart it. Its
  offsets are committed only after the DB write and inserts are idempotent on `(job_id, seq)`,
  so the history tables should end with every `seq` exactly once.

## 6. Results

All values in this section are placeholders until the runs are done.

### 6.1 Anomalies per run

`TODO(measure)`: one row per job, read from `jobs` and `pose_history`/`training_history`.

| Job | Poses | Snapshots | Pose anomalies | Training anomalies | Injected | Dropped |
|---|---|---|---|---|---|---|
| TODO(measure) | | | | | | |

### 6.2 Contamination results

`TODO(measure)`: output of `experiment.py`.

| contamination | shuffle precision | shuffle recall | blur precision | blur recall | clean FP rate | F1 |
|---|---|---|---|---|---|---|
| 0.01 | TODO(measure) | | | | | |
| 0.05 | TODO(measure) | | | | | |
| 0.10 | TODO(measure) | | | | | |

### 6.3 Training throughput with streaming on and off

`TODO(measure)`: output of `bench_overhead.py`, same scene and seed, at least 2k steps.

| Streaming | steps/s (mean ± std) | Δ |
|---|---|---|
| off | TODO(measure) | |
| on | TODO(measure) | TODO(measure) |

### 6.4 Health check

`TODO(measure)`: paste the output of two consecutive `health.py` runs here, showing the counts
climbing and anomalies > 0.

```
TODO(measure)
```

### 6.5 Processor restart without loss

`TODO(measure)`: kill the processor at seq ≈ N, restart it, then record:
- the max `seq` and row count per table compared with the producer's sent count
- any duplicate or missing `seq`
- the time until the dashboard caught up

| Table | Producer sent | Rows after restart | Missing seq | Duplicates | Catch-up time |
|---|---|---|---|---|---|
| pose_history | TODO(measure) | | | | |
| training_history | TODO(measure) | | | | |

## 7. Discussion

- **What Isolation Forest can and cannot see.** It scores one vector at a time, so it detects
  sudden departures: jumps, flips, confidence drops, loss spikes. Slow drift that stays within
  the window's range is harder to catch. The window-relative features (`conf_ratio`,
  `focal_ratio`, `loss_ratio_best`, `loss_trend`) make the vectors partly scale-free across
  scenes. A scene unlike anything in the training runs can still raise false positives.
- **Contamination is a threshold, not a rate estimate.** A fixed contamination flags about that
  fraction of normal-looking windows on clean data. The experiment in 5.4 exists to pick a value
  for this pipeline instead of trusting the sklearn default.
- **Unscored sources.** Excluding `reloc` and `map` bursts avoids false positives from expected
  refits. It also means that a bad refit is not flagged by the pose model. It still shows on the
  camera path and, indirectly, in the training metrics.
- **Dropping instead of blocking.** Keeping training fast means the monitor can lose events
  under load or while the broker is down. Losses are counted and shown, and `seq` gaps make them
  visible per job. They are never silent.
- **Logging interval.** The trainer hook can emit only when the trainer calls `add_scalar`.
  `simple_trainer` defaults to `tb_every = 100`, so live runs must set it to 10 to meet the
  10-step pace.
- **Scale.** A single partition per topic and a single processor are enough at a few events per
  second. The growth plan in architecture.md covers the database side.

## 8. Conclusion

The streaming path watches poses and training while the run is in progress and leaves the batch
reconstruction unchanged. Hooks are attached at runtime, sends never block, and the file
hand-off remains as a fallback. Bad events are dropped and counted, the models are trained
offline and only scored in the stream, and a restarted processor resumes from committed offsets
without duplicates. The open items are the measurements in section 6: the contamination choice,
the steps/s overhead and the restart demonstration.
