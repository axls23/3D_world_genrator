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

The browser opens `http://localhost:8090`: the viser 3DGS viewer (port 8080) on the left once
training starts, the camera feed on the right (keyframes flash green) with bootstrap guidance,
training progress and stream counters. Before the first map the left pane shows what the
pipeline needs ("move the camera slowly: 12/24 keyframes"); with `--depth` it also shows the
live depth of the camera (~3 fps) while the first ACE map is built. The dashboard binds to localhost
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

```bash
# Recorded video, just in time (frames as fast as ACE keeps up)
python scripts/pipeline/automated_intelligent_pipeline.py demo/input/drone_demo.mp4 \
    --stream --difix-online --max_steps 7000 --output_dir demo/output/stream

# Rehearse the live path on a recording (a 30 fps "camera" that drops frames when busy)
... drone_demo.mp4 --stream --stream-pace realtime --difix-online

# Live sources (run until --max_steps, default 10000 - a hard ceiling - or a graceful stop)
... 0 --stream                          # first local camera
... /dev/video2 --stream
... rtsp://192.168.1.20:8554/cam --stream
... flight://0.0.0.0:8815 --stream      # wait for an external camera over Arrow Flight
```

Stop a live run gracefully (final checkpoint, PLY and eval are still written):
`touch <output_dir>/results/stream_3dgs/STOP` or `POST /api/stream/stop` (only a STOP written
after the run started counts; it is honoured during the bootstrap too). Viewer / dashboard:
`--stream-viewer-port 8080 --stream-dashboard-port 8090`.

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

- `--depth` (demo) / `--stream-depth`: Depth Anything V2 Small on each registered keyframe,
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
- The first map needs `stream_bootstrap_frames` (24) keyframes with some parallax, then
  ~70 s of ACE training on this GPU before training starts.
- One camera (single intrinsics) per stream.
