# Streaming reconstruction (video file → live camera)

The streaming pipeline (`hypersplat/stream/`) trains 3DGS while frames arrive, instead of
extracting frames → running ACE-Zero → training → refining in separate batch stages. The same
code reconstructs a recorded video *just in time* and a live camera *in real time*.

```
camera / video / Flight producer
   │  Frame (RGB, ≤ 960 px)
   ▼
KeyframeGate ── drops blurry + redundant frames (485 frames → 28 keyframes on the drone clip)
   ▼
KeyframeStore ── JPEG in Arrow buffers, hard budget (256 MB / 400 keyframes), redundancy eviction
   │  ACE-res copy on tmpfs
   ▼
ACE sidecar (ace0 env) ── bootstrap map → localize each keyframe (~35 ms) → refit in background
   │  Arrow IPC: poses + seed points
   ▼
simple_trainer.py ── samples live Arrow snapshots, grows Gaussians into new space,
                     online Difix rounds replace their previous pseudo views
```

Nothing is written to disk except the trainer's checkpoints/PLY; the ACE frames and map live
in `/dev/shm` and are removed on exit.

## Demo script (live camera + split view)

```bash
demo/run_stream_demo.sh                      # external USB camera, opens the split view in the browser
demo/run_stream_demo.sh --record-seconds 90  # stop recording after 90 s, then finish training
demo/run_stream_demo.sh --camera /dev/video2
demo/run_stream_demo.sh --builtin            # the laptop's own camera
demo/run_stream_demo.sh --video --realtime   # the drone clip played like a live camera
demo/run_stream_demo.sh --help
```

The default source is an **external USB camera** (a UVC camera on a non-fixed USB port; the
laptop's built-in camera is never picked automatically), else an Android phone on the USB
cable (webcam mode or USB debugging). If neither is connected the script asks you to plug one
in and waits, printing what is missing (e.g. an unaccepted USB-debugging prompt); without a
terminal it exits instead. `--camera N` / `--builtin` choose a camera explicitly.

### Defaults of the demo script

`demo/run_stream_demo.sh` composes the stream defaults; the pipeline and trainer keep stock
defaults (no 3DGUT, cold start, L-BFGS or Difix), so the script is the place to change the demo.
Every run prints the effective set on start (`config :` and `command:` lines).

| Default | Setting | Opt out |
|---|---|---|
| Cold start (no ACE bootstrap wait, continuous ACE) | on | `--no-cold-start`; off automatically with `--map-cache` / `--replay` |
| Mapper depth prior (weight 50) | on | `--no-mapper-depth` |
| 3DGUT (`--with_ut`) | on | `--no-ut` (`--refine-poses` also turns it off) |
| L-BFGS polish every 50 steps | on | `--no-lbfgs` |
| Online Difix (share 0.15, from 50 % of the run) | on | `--no-difix` |
| Seed 42 | on | `--seed N` |
| Step ceiling | auto: 7000 video / 10000 live | `--steps N` |
| V-JEPA 2 latent loss: ViT-L is the main latent model; Depth Anything / DINOv2 features are its secondary stream, hooked into the predictor | off | on with `--jepa`; tune with `--jepa-lambda/-warmup/-anneal-end`, the secondary stream with `--jepa-backend/-layer/-side`. Needs the ViT-L checkpoint ([vitl.pt](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt)) at `/mnt/Shared/3D_WORLD_GENRATION/hf_cache/vjepa2/vitl.pt` (a path in `$HYPERSPLAT_VJEPA_CKPT` replaces that default); a missing file stops the trainer at startup. Plain feature loss: `--trainer-arg latent_model=feature` (pipeline) |
| MPS, refit recipe, gauge lock, ring, device poses, record | off | `--mps`, `--refit-recipe ace0`, `--gauge-lock`, `--cuda-ring`, `--device-poses`, `--record` |

Evidence behind these defaults, with the status of each number:

| Setting | Measured | Status |
|---|---|---|
| Mapper depth, hard prior (ACE depth loss) | inliers 84 % -> 43 %, -2.3 dB | measured; not the default prior |
| Mapper depth, soft prior (the default) | `ab_cold_depth`: -2.27 dB (n=1); `ab_cold_depth2`: 23.40 vs 23.20 dB, but LPIPS 0.501 vs 0.431 (n=2) | mixed; unverified |
| Online Difix | no results file in the repo for the drone 2k A/B | unverified |
| L-BFGS polish | +0.003 dB on the 2k drone run | within noise |
| 3DGUT | the `--with_ut` help says "same quality as standard" on the drone A/B; no stream-demo result file | unverified for the demo |
| Deterministic mode | ~0.2 dB lower (help text); no results file | unverified |

Changing a default should be backed by an A/B run under `experiments/` before it lands here.

**Stop recording.** A live source otherwise keeps adding frames for as long as it runs. Press
`s` in the terminal, click *Stop recording* in the dashboard, `touch
<output_dir>/results/stream_3dgs/STOP_RECORDING`, or set `--record-seconds S` /
`--record-keyframes K`: the camera is released, the last map refit runs, and training finishes
on the keyframes collected (like a video file that ended). Ctrl+C (`STOP`) ends training
itself.

**Early stop when the world is reconstructed.** With early stopping on (the pipeline default;
`--no-auto-stop` / `--no-early-stopping` turns it off) `max_steps` is only a ceiling. Every
500 steps the trainer probes the model (`[WorldStop]` log lines, the *World* line in the
dashboard, `stream/probe_psnr` in TensorBoard) and ends training once, on two consecutive
probes:

| signal   | condition |
|----------|-----------|
| coverage | no new registered views for 1500 steps and the ACE map idle (no refit, nothing pending) |
| quality  | held-out PSNR (up to 6 held-out keyframes, full resolution) gained < 0.15 dB over 3 probes, or is above the target (`--target-psnr`, default 22 dB) and gains < 0.5 dB; without held-out views the EMA training loss is used |
| growth   | Gaussian count changed < 2 % over the same window (default strategy only: MCMC grows to its cap on a schedule, so it is ignored there) |
| minimum  | at least 3000 steps |

New views reset the window (the scene just grew). When it stops, recording is stopped too and
the final checkpoint, PLY and eval are written as usual. With `stream_cycle` off and online
Difix on, convergence first triggers one final polish round (in chunks, training continues),
then `difix_final_tail_steps` (750) more steps on its pseudo views before the end. Tuning: `--stream_converge_*` options
of `simple_trainer.py` (logic in `examples/trainer_schedule.py`, `WorldConvergence`).

The browser opens `http://localhost:8090`: the viser 3DGS viewer (port 8091) on the left once
training starts, the camera feed on the right (keyframes flash green) with bootstrap guidance,
training progress and stream counters. Before the first map the left pane shows what the
pipeline needs ("move the camera slowly: 12/24 keyframes"); with `--depth` it also shows the
live depth of the camera (~3 fps) while the first ACE map is built.
The header's "Streaming view" link opens `/streaming`, a telemetry view of the same run (KPIs, pose map, camera feed
with an optional depth heatmap overlay); the base page itself is unchanged. The dashboard binds to localhost
only; `--share` exposes it (and the camera feed) on the network. Ctrl+C once stops gracefully
(final checkpoint + PLY), twice kills. `demo/run_demo.sh` (the web app) is unchanged.

### Android phone as the camera (USB-C / Thunderbolt port)

Phones do not send video over Thunderbolt, but the port is also USB-C, so the cable works:

```bash
demo/run_stream_demo.sh --phone                      # back camera
demo/run_stream_demo.sh --phone --phone-camera front
```

`--phone` (detection in `hypersplat/stream/phone.py`) uses, in order:

1. **Android 14+ USB webcam mode** - plug in, tap the USB notification, choose *Webcam*. The
   phone is a normal UVC camera; nothing to install. Found by its Android vendor id / external
   USB port (built-in cameras sit on fixed ports).
2. **scrcpy** (any Android 12+, back camera, over USB debugging) streaming into a
   v4l2loopback device; the script starts and stops scrcpy itself. One-time setup:

   ```bash
   sudo dnf install android-tools akmod-v4l2loopback           # RPM Fusion
   sudo dnf copr enable zeno/scrcpy && sudo dnf install scrcpy  # scrcpy >= 2.2
   sudo modprobe v4l2loopback video_nr=10 card_label=phone exclusive_caps=1
   ```
   Then enable *Developer options > USB debugging* and accept the prompt on the phone.

Without either, `--phone` prints exactly what is missing. Over Wi-Fi instead, any IP-camera
app works as a URL source (`demo/run_stream_demo.sh http://PHONE_IP:8080/video`). For good
reconstructions walk around the subject, stay on the 1x lens (no zoom), turn stabilization off
and lock focus/exposure if the app allows: the pipeline assumes one fixed camera per stream.

Endpoints: `/` split page, `/frame.jpg` newest frame, `/feed.mjpg` MJPEG (VLC etc.),
`/status.json` counters.

## Run it

Native (no Docker). The script sets `HF_HOME` to `../hf_cache` when it exists; the trainer
needs the repo root on `PYTHONPATH` (the script sets it for the trainer only).

```bash
# Final demo, recorded video, with the default set above (the command the script prints)
demo/run_stream_demo.sh --video

# Same, 2000 steps (a quick check, not a quality run)
demo/run_stream_demo.sh --video --steps 2000

# Rehearse the live path on a recording (a 30 fps "camera" that drops frames when busy)
demo/run_stream_demo.sh --video --realtime

# Live sources: the script's defaults apply; --steps is a ceiling (default 10000)
demo/run_stream_demo.sh --camera /dev/video2
demo/run_stream_demo.sh rtsp://192.168.1.20:8554/cam
demo/run_stream_demo.sh flight://0.0.0.0:8815
```

Calling the pipeline directly gives stock defaults, not the demo set. Add the flags explicitly
if you need them (`--stream --difix-online --with_ut --stream-depth ...`); see
`demo/run_stream_demo.sh --help` for the trainer-arg equivalents.

Stop a live run gracefully (final checkpoint, PLY and eval are still written):
`touch <output_dir>/results/stream_3dgs/STOP` or `POST /api/stream/stop` (only a STOP written
after the run started counts; it is honoured during the bootstrap too). Viewer / dashboard:
`--stream-viewer-port 8091 --stream-dashboard-port 8090`.

### Through Apache Kafka (`--streaming`)

`demo/run_stream_demo.sh --streaming ...` (or `run_stream_demo.bat --streaming`) sends the run's camera
poses and training metrics to Apache Kafka with no changes to the pipeline: the hooks in
`hypersplat/streaming` wrap the stream session and the trainer's TensorBoard writer. Start the data layers
first with `hypersplat/streaming/ops/pod.sh up` (Kafka, Postgres, processor, Streamlit on 8501). The
processor scores every event; the 8090 page's **Streaming visualization** view shows the flags for this
run's job (`HYPERSPLAT_JOB_ID`) as they arrive. If Kafka is down events are dropped, never blocking training.

### External cameras over Arrow Flight

On the machine with the camera (only `pyarrow` + `opencv-python` needed):

```bash
python -m hypersplat.stream.flight push 0 grpc://gpu-box:8815 --fps 10     # camera 0
python -m hypersplat.stream.flight get grpc://gpu-box:8815 stats           # live counters
python -m hypersplat.stream.flight get grpc://gpu-box:8815 views           # poses, no pixels
```

Frames are JPEG-encoded at the producer (~1 MB/s at 960 px, 10 fps). Any source can also
publish `views`/`stats` with `--stream-flight-port 8815`.

### API server

`POST /api/jobs/submit` with `{"stream_source": "0", "difix_online": true}` (optional
`stream_pace`, `stream_budget_mb`, `stream_flight_port`).

## Online Difix (also for batch runs)

`--difix-online` runs Difix rounds *inside* one training run instead of STEP 3's
render → Difix → retrain-from-scratch loop. Each round renders pseudo views from the live
model, repairs them with `nvidia/difix_ref`, weights them by source trust × angle falloff ×
change-PSNR trust, and replaces the previous round's views. The orbit angle and base weight
follow the same closed-loop strategies as the batch loop; rounds stop when a round shows
hallucination. Batch runs: rounds at 40/60/80 % of `max_steps`.

**Streams train in cycles** (`stream_cycle`, default on): after every epoch - one pass over the
current train keyframes, at least `stream_epoch_min_steps` (100) steps - the trainer runs

```
[train one epoch] -> [Difix: 2 views at scale 0.5 into a rolling pool of 16 pseudo views]
                  -> [depth: re-align the maps of keyframes ACE re-localized (cached network output)]
                  -> [ACE: start a background refit if >= 4 keyframes changed since the last map]
```

The refit runs in the sidecar while training continues; its poses reach the next epoch's
depth re-alignment. While a refit holds the GPU lease an epoch's Difix step is skipped (logged).
`[Cycle]` lines log each step's cost. Tuning: `difix_epoch_views`, `difix_epoch_scale`,
`stream_cycle_refit_min`, `stream_epoch_min_steps`; `--stream_cycle False` restores the
schedule below.

With `stream_cycle` off, streams never freeze for a whole round: a round renders + repairs views in chunks of up to
`difix_chunk_s` (3 s), one chunk every `difix_chunk_every` (50) steps, holding the GPU lease
only per chunk; at most `difix_online_views` (16) pseudo views per round, starting at Difix
scale 0.75 when less than `difix_low_vram_mb` (3.5 GB) is free. With the world early stop a
stream runs one round at `difix_online_start` (1500) plus the final polish round; without it,
every `difix_online_every` (2000) steps. Each round logs Difix / render ms per view, chunk
count, longest chunk and wall time (also in `scene_profile.json` under `difix.rounds`).

## Dense depth, pose and exposure refinement (streams)

- Depth is on in the demo through the mapper prior (`--no-mapper-depth` turns that off; see the
  defaults table for the evidence). `--depth` (demo) / `--stream-depth` (pipeline, opt-in):
  Depth Anything V2 Small on each registered keyframe,
  sharpened with a guided filter, aligned to the keyframe's ACE points (robust affine fit in
  inverse depth, scored on held-out points), edge pixels dropped. Seeds new Gaussians densely
  (until 80 % of the Gaussian cap) and adds a log-depth loss. Drone clip @7k: 23.99 -> 24.99 dB,
  LPIPS 0.333 -> 0.292; ~40 ms per keyframe.
- `--refine-poses` (demo) / `--stream-pose-opt --stream-exposure-opt`: per-keyframe pose
  deltas and exposure gain + bias, in slots that follow keyframe ids (reused after eviction,
  pose reset when an ACE refit moves the keyframe). Exposure is applied relative to the mean
  over live keyframes, so the model keeps a neutral exposure and held-out renders have no
  colour cast. Pseudo and held-out views are never corrected.
- `--stream-bootstrap-cooldown N`: cap the first ACE map's cooldown iterations (sooner first
  map; background refits recover accuracy).

## ACE refits: recipe, gauge lock, continuous mapping (opt-in)

The refit recipe, gauge lock and continuous mapping are not A/B-tested on 3DGS quality.
Continuous mapping is on by default in the demo (through cold start); the other two are opt-in
(`--refit-recipe ace0`, `--gauge-lock`). Measurements are from
`scripts/experiments/ace_refit_bench.py` (the ACE loop alone; Kenilworth clip, 400 keyframes,
cached bootstrap map).

- `--refit-recipe ace0`: refits train like an ACE-Zero mapping iteration: 1cyclepoly warmup to
  3e-3, early cooldown once 70 % of reprojection errors are under 10 px, pose MLP starting at
  the input poses (zero-init output layer, `--pose_refinement_zero_init` in train_ace). Refits
  take ~70-115 s instead of 100-260 s.
- `--gauge-lock`: ACE-Zero reconstructs up to a similarity, and every refit rotates and scales
  the whole map a little (0.3-1 deg and ~1 % per refit on Kenilworth). The lock fits a robust
  Sim(3) (RANSAC over frame pairs, never worse than identity or the previous lock;
  `hypersplat/stream/gauge.py`) from each refit's poses to the trainer's and applies it, so
  the trainer's frame stays the bootstrap's. Each refit logs `Refit vN gauge: ...` with the
  movement left after alignment.
- `--ace-continuous`: no refits; after the first map the ACE head trains a few iterations per
  3DGS step and keyframes are relocalized round-robin (`scripts/acezero/stream_mapper.py`).
  Experimental: in its first tuning it registered fewer frames (208/400 vs ~360 after 10 min)
  and let poses drift more than refits, at +80-110 ms per 3DGS step.
- Known: the biggest pose jumps after a refit are frames registered by the previous refit (up
  to 34 % of the path); older frames move 0.1-0.3 %. Neither option above changes that.

## Viewer walk mode

Stream runs open the viewer in a ground-level walk mode: WASD walk, drag to look, Shift run,
pitch limited to 35 deg, the camera kept inside the region the training cameras covered and at
their height (it grows as keyframes register). "Clean view" hides near-transparent and
needle-shaped splats (viewer only), and moving frames never drop below "Min move res".
`--no-walk` restores free-fly. Walk mode needs the patched viser client:
`scripts/viewer/build_walk_client.sh` (`--restore` puts the stock client back); the demo warns
if it is missing. A finished run writes `walk_bounds.npz`; `examples/simple_viewer.py
--walk_bounds <file>` walks it again.

## Memory on a 6 GB GPU

| Component | VRAM |
|---|---|
| 3DGS training at 960×540 | ~1.5–3.7 GB peak |
| ACE sidecar (context + network) | ~0.4 GB |
| Difix (UNet + VAE; text encoder dropped) | 1.8 GB weights + 1.6 GB activations at 960×540 |

Difix is parked on the CPU between rounds. On a CUDA OOM it steps down its working
resolution (1.0 → 0.85 → 0.75 → 0.6 → 0.5 ×) and remembers what fit; if nothing fits the
round is skipped and training continues. Other GPU processes (e.g. an open viewer) count
against the same 6 GB.

## Native code

`hypersplat/stream/csrc/streamops.c` (built on first use with the system C compiler, loaded
via ctypes, numpy fallback without a compiler) holds the Python-level hot paths:

| Path | Python | C |
|---|---|---|
| Eviction redundancy (pose-space nearest neighbour, 400 keyframes) | 1152 ms | 2 ms |
| Weighted view draw per training step | — | µs |

JPEG decode (libjpeg) is overlapped with the GPU step on a prefetch thread, so the trainer
waits ~0.2 ms per step for data. Rasterization (gsplat CUDA), RANSAC (dsacstar C++) and the
image ops (OpenCV) were already native.

## Limits

- Per-image modules (`--pose_opt`, `--app_opt`, bilateral grid, masks, depth loss) are
  disabled for streams: they are sized once while the keyframe set keeps changing. ACE
  refits refine poses instead.
- Without cold start (`--no-cold-start`, or with `--map-cache` / `--replay`), the first map
  needs `stream_bootstrap_frames` (24) keyframes with some parallax, then ~70 s of ACE training
  on this GPU before training starts. With the demo default (cold start) the first keyframe's
  depth seeds the map within seconds instead.
- One camera (single intrinsics) per stream.
