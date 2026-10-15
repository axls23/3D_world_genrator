#!/usr/bin/env bash
# Live streaming 3DGS demo: camera -> keyframes -> ACE sidecar -> live training + online Difix,
# with a split view in the browser: 3DGS viewer (viser) left, camera feed right.
# See docs/getting-started/streaming.md.
#
#   ./run_stream_demo.sh                               # external USB camera (default, auto-detected)
#   ./run_stream_demo.sh --camera 2                    # a specific camera index / /dev/videoN
#   ./run_stream_demo.sh --builtin                     # the laptop's own camera
#   ./run_stream_demo.sh --phone                       # Android phone over USB-C (webcam mode / scrcpy)
#   ./run_stream_demo.sh --wifi                        # find an IP camera / phone app on the Wi-Fi
#   ./run_stream_demo.sh rtsp://192.168.1.20:8554/cam  # network camera
#   ./run_stream_demo.sh flight://0.0.0.0:8815         # remote camera pushing over Arrow Flight
#   ./run_stream_demo.sh --video                       # recorded clip (demo/input/drone_demo.mp4)
#   ./run_stream_demo.sh --video my.mp4 --realtime     # a recording played like a live camera
#
# s (terminal) or "Stop recording" (browser): camera off, training finishes on what was captured.
# Ctrl+C once: graceful stop (final checkpoint, PLY and eval are written). Twice: kill.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --camera N|DEV|usb  camera index, /dev/videoN, or usb (default: first external USB camera;
                      the laptop's built-in camera is never picked automatically)
  --builtin           use the laptop's built-in camera (same as --camera 0)
  --phone             Android phone on the USB-C / Thunderbolt port: USB webcam mode if the phone
                      offers it, else its camera over adb (USB debugging; no app, no sudo)
  --wifi              scan the local network for an IP camera / phone camera app and use it
                      (several found: asks which; python -m hypersplat.stream.wifi lists them)
  --phone-camera C    back (default) or front
  --fps N             capture frame rate asked of a USB phone or camera (default 60; a Wi-Fi
                      phone app sets its own rate in the app)
  --video [FILE]      use a video file instead of a camera (default demo/input/drone_demo.mp4)
  --realtime          play a video like a live camera (frames drop when busy)
  --record-seconds S  stop recording after S seconds (training continues on what was captured)
  --record-keyframes K
                      stop recording after K keyframes (training continues)
  --steps N           training step ceiling (default 7000 for videos; live sources have none:
                      they start at 2000 and grow ~1000 past each new view, like a download ETA);
                      training stops earlier once the world is reconstructed
  --target-psnr DB    held-out PSNR that counts as reconstructed once gains flatten (default 22)
  --no-auto-stop      train to the step ceiling instead of stopping on convergence
  --depth             dense depth per keyframe (Depth Anything V2, aligned to ACE points):
                      denser seeding + depth loss, ~20 ms per keyframe
  --depth-feedback K  implies --depth; every epoch also runs Depth Anything on the 3DGS render of K
                      keyframes next to the camera frame and down-weights the depth loss where they
                      disagree (alongside the ACE refit and training; K=3 is cheap)
  --refine-poses      refine each keyframe's pose and exposure while training (uses the standard
                      rasterizer: 3DGUT has no pose gradients, so this turns --ut off)
  --ut / --no-ut      3DGUT: unscented-transform projection + world-space (eval3d) rasterizer
                      (default on; same quality as the standard rasterizer on the drone A/B,
                      handles distorted / rolling-shutter cameras)
  --map-cache DIR     videos only: keep the first ACE map in DIR and reuse it when the same
                      file is run again (no bootstrap wait, same poses run to run)
  --no-difix          disable online Difix refinement
  --difix-share F     cap on the Difix pseudo views' share of training draws (default 0.15;
                      uncapped they took ~40% of the steps); none = no cap
  --difix-start F     online Difix starts at this fraction of the run (default 0.5: the early
                      blob renders are what Difix hallucinates on); 0 = from the start
  --seed N            training random seed (default 42)
  --refit-recipe R    ACE refit training: legacy (default) or ace0 (the ACE-Zero paper's mapping
                      iteration: warmup to 3e-3, early cooldown on the inlier trigger, pose MLP
                      starting at the input poses)
  --gauge-lock        align every ACE refit to the bootstrap map (Sim(3)) before its poses reach
                      the trainer: removes the map's per-refit rotation / scale creep
  --no-walk           viewer: free-fly camera instead of the ground-level walk mode
  --ace-continuous    ACE maps in step with training: after the first map the ACE head trains a
                      little every 3DGS step and keyframes are relocalized one at a time (settled
                      ones are anchored), instead of ~2 min refits landing as pose jumps
  --cold-start / --no-cold-start
                      no ACE bootstrap wait (implies --ace-continuous): the first keyframe's
                      Depth Anything depth seeds the ACE map in seconds, training starts on it,
                      and the following keyframes register as they arrive. ON by default, except
                      with --map-cache / --replay (they need a bootstrap map)
  --focal F           with --cold-start: focal length in px of the ACE-res (480 short side)
                      frames; default ACE's guess (0.7 x diagonal), which is not refined
  --mapper-depth [W] / --no-mapper-depth
                      ON by default (with cold start / --ace-continuous): Depth Anything keeps guiding the ACE
                      mapper after the seed. Each registered keyframe's aligned depth becomes a soft
                      scene-coordinate prior under its current pose (weight W, default 50; the
                      reprojection loss stays the main loss). Implies --depth. Experimental: a hard
                      version of this prior (ACE's own depth loss) dropped the mapper's inliers
                      from 84% to 43% and cost 2.3 dB; the soft one is unverified: 1 run at 23.40 dB vs 23.20 plain (spread ~1.3 dB)
  --lbfgs [EVERY] / --no-lbfgs
                      second-order polish (ON by default, every 50 steps): after densification stops, every EVERY steps (default
                      50) an L-BFGS round (strong-Wolfe line search, 4 fixed views) on all Gaussian
                      parameters; a round that does not lower its batch loss is reverted. Measured
                      +0.003 dB on the 2k drone run (noise) - it needs a long refinement tail
  --deterministic     reproducible runs of a video file: deterministic kernels for training and
                      ACE maps, file fully localized before step 0 (slower; ~0.2 dB lower)
  --no-view           no viewer / dashboard / browser
  --no-browser        start the dashboard but do not open a browser
  --viewer-port P     viser 3DGS viewer port (default 8080)
  --dashboard-port P  split-view dashboard port (default 8090)
  --share             serve the dashboard on all interfaces (exposes the camera feed!)
  --budget-mb MB      keyframe store memory budget (default 256)
  --cuda-ring         with --ace-continuous: ACE pose rows reach the trainer through shared-memory
                      rings (fixed-size records) instead of the Arrow pipe
  --mps [T,S[,MEM]]   CUDA MPS: trainer and ACE sidecar share the GPU concurrently instead of
                      time-slicing it; T / S = SM percentage caps (default 100,100), MEM = the
                      sidecar's GPU memory cap (e.g. 1500M; it backs off instead of the trainer
                      running out). Starts a private MPS daemon for this run, stopped on exit
  --device-poses      ACE poses applied on the GPU by a CUDA kernel (device pose table); real
                      views render with the table's pose (same poses as without it)
  --pose-filter A     with device poses: filter relocalization jitter, gain A in (0, 1]
                      (0.3 averages ~5 relocalizations; corrections pass at once)
  --tiers             with device poses: sample views by pose stability (provisional x0.3)
  --record            write the ACE events this run applied (with the training step) to
                      <output>/results/stream_3dgs/stream_events.arrows
  --replay FILE       replay such a recording instead of running ACE: the same pose stream,
                      step for step (use the same video, --map-cache and options)
  --profile-cuda N[@S] CUDA timeline of N training steps from step S (default 500): Chrome
                      trace + sync/copy/kernel summary in <output>/results/stream_3dgs/profile
  --jepa / --no-jepa  latent (JEPA-style) loss: 1 - cos between frozen-ViT patch features of the
                      render and the photo, added to the photometric loss. OFF by default.
                      Feature model: Depth Anything V2 backbone (default) or plain DINOv2
                      (note: this is not V-JEPA; it failed its 300-step gate on the drone clip)
  --jepa-lambda L     latent loss weight (default 0.1)
  --jepa-warmup S     no latent loss before step S (default 0)
  --jepa-anneal-end S linearly anneal the weight to 0 at step S (default 0: hold constant)
  --jepa-backend B    depth_anything (default) or dinov2
  --jepa-layer L      ViT layer: -1 last (default); depth_anything accepts 3, 6, 9, 12
  --jepa-side PX      long side fed to the ViT (default 224)
  --adaptive-spacing  measure the rotation between adjacent keyframes (per-source spacing; logs only
                      until a consumer uses it; off by default)
  --debug-offered DIR save every offered frame (ACE res) and its gate decision to DIR (offers.jsonl)
  --output DIR        output directory (default demo/output/stream_demo)
  -h, --help          this help
EOF
}

SOURCE="usb"
OUTPUT="demo/output/stream_demo"
DEMO_VIDEO="demo/input/drone_demo.mp4"
EXTRA=()
DIFIX=1 VIEW=1 BROWSER=1 PHONE=0 WIFI=0 PHONE_FACING=back CAPTURE_FPS=60
DIFIX_SHARE=0.15 DIFIX_START=0.5 SEED=42
UT=default REFINE_POSES=0
COLD=default LBFGS=1 LBFGS_EVERY=50 MAPDEPTH=1 MAPDEPTH_W=50
JEPA=0 JEPA_TUNED=0 JEPA_LAMBDA=0.1 JEPA_WARMUP=0 JEPA_ANNEAL=0
JEPA_BACKEND=depth_anything JEPA_LAYER=-1 JEPA_SIDE=224
VIEWER_PORT=8080 DASH_PORT=8090 DASH_HOST=127.0.0.1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera)         SOURCE="$2"; shift ;;
        --builtin)        SOURCE="0" ;;
        --phone)          PHONE=1 ;;
        --wifi)           WIFI=1 ;;
        --phone-camera)   PHONE_FACING="$2"; shift ;;
        --fps)            CAPTURE_FPS="$2"; shift ;;
        --video)          if [[ $# -gt 1 && "$2" != -* ]]; then SOURCE="$2"; shift; else SOURCE="$DEMO_VIDEO"; fi ;;
        --realtime)       EXTRA+=(--stream-pace realtime) ;;
        --steps)          EXTRA+=(--max_steps "$2"); EXTRA_STEPS="$2"; shift ;;
        --target-psnr)    EXTRA+=(--stream-target-psnr "$2"); shift ;;
        --no-auto-stop)   EXTRA+=(--no-early-stopping) ;;
        --record-seconds) EXTRA+=(--stream-record-seconds "$2"); shift ;;
        --record-keyframes) EXTRA+=(--stream-record-keyframes "$2"); shift ;;
        --depth)          EXTRA+=(--stream-depth) ;;
        --depth-feedback) EXTRA+=(--stream-depth --trainer-arg "stream_depth_feedback=$2"); shift ;;
        --refine-poses)   EXTRA+=(--stream-pose-opt --stream-exposure-opt); REFINE_POSES=1 ;;
        --ut)             UT=1 ;;
        --no-ut)          UT=0 ;;
        --map-cache)      EXTRA+=(--stream-map-cache "$2"); shift ;;
        --no-difix)       DIFIX=0 ;;
        --difix-share)    DIFIX_SHARE="$2"; shift ;;
        --difix-start)    DIFIX_START="$2"; shift ;;
        --seed)           SEED="$2"; shift ;;
        --deterministic)  EXTRA+=(--deterministic) ;;
        --ace-continuous) EXTRA+=(--trainer-arg "stream_ace_mode=continuous") ;;
        --cold-start)     COLD=1 ;;
        --no-cold-start)  COLD=0 ;;
        --focal)          EXTRA+=(--trainer-arg "stream_cold_focal=$2"); shift ;;
        --mapper-depth)   MAPDEPTH=1
                          if [[ "${2:-}" =~ ^[0-9.]+$ ]]; then MAPDEPTH_W="$2"; shift; fi ;;
        --no-mapper-depth) MAPDEPTH=0 ;;
        --lbfgs)          LBFGS=1
                          if [[ "${2:-}" =~ ^[0-9]+$ ]]; then LBFGS_EVERY="$2"; shift; fi ;;
        --no-lbfgs)       LBFGS=0 ;;
        --refit-recipe)   EXTRA+=(--trainer-arg "stream_refit_recipe=$2"); shift ;;
        --gauge-lock)     EXTRA+=(--trainer-arg "stream_gauge_lock=true") ;;
        --no-walk)        EXTRA+=(--trainer-arg "viewer_walk=false") ;;
        --no-view)        VIEW=0 ;;
        --no-browser)     BROWSER=0 ;;
        --viewer-port)    VIEWER_PORT="$2"; shift ;;
        --dashboard-port) DASH_PORT="$2"; shift ;;
        --share)          DASH_HOST=0.0.0.0 ;;
        --budget-mb)      EXTRA+=(--stream-budget-mb "$2"); shift ;;
        --cuda-ring)      EXTRA+=(--trainer-arg "stream_ring=true") ;;
        --mps)            if [[ $# -gt 1 && "$2" =~ ^[0-9] ]]; then MPS="$2"; shift; else MPS="100,100"; fi ;;
        --record)         EXTRA+=(--trainer-arg "stream_record=true") ;;
        --device-poses)   EXTRA+=(--trainer-arg "stream_device_poses=true") ;;
        --pose-filter)    EXTRA+=(--trainer-arg "stream_device_poses=true" --trainer-arg "stream_pose_alpha=$2"); shift ;;
        --tiers)          EXTRA+=(--trainer-arg "stream_device_poses=true" --trainer-arg "stream_tiers=true") ;;
        --replay)         EXTRA+=(--trainer-arg "stream_replay=$(realpath "$2")"); shift ;;
        --profile-cuda)   EXTRA+=(--trainer-arg "profile_cuda_steps=${2%@*}")
                          [[ "$2" == *@* ]] && EXTRA+=(--trainer-arg "profile_cuda_start=${2#*@}")
                          shift ;;
        --jepa)           JEPA=1 ;;
        --no-jepa)        JEPA=0 ;;
        --jepa-lambda)    JEPA_TUNED=1; JEPA_LAMBDA="$2"; shift ;;
        --jepa-warmup)    JEPA_TUNED=1; JEPA_WARMUP="$2"; shift ;;
        --jepa-anneal-end) JEPA_TUNED=1; JEPA_ANNEAL="$2"; shift ;;
        --jepa-backend)   JEPA_TUNED=1; JEPA_BACKEND="$2"; shift ;;
        --jepa-layer)     JEPA_TUNED=1; JEPA_LAYER="$2"; shift ;;
        --jepa-side)      JEPA_TUNED=1; JEPA_SIDE="$2"; shift ;;
        --adaptive-spacing) EXTRA+=(--trainer-arg "stream_adaptive_spacing=true") ;;
        --debug-offered)  EXTRA+=(--trainer-arg "stream_debug_offered_dir=$(realpath -m "$2")"); shift ;;
        --output)         OUTPUT="$2"; shift ;;
        -h|--help)        usage; exit 0 ;;
        -*)               echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                SOURCE="$1" ;;
    esac
    shift
done
EXTRA+=(--seed "$SEED")
# Cold start (no ACE bootstrap wait) by default, unless the run needs a bootstrap map: a cached or
# recorded map, or a replay. An explicit --cold-start with those is the user's call.
if [[ "$COLD" == default ]]; then
    COLD=1
    for a in "${EXTRA[@]}"; do [[ "$a" == --stream-map-cache || "$a" == stream_replay=* ]] && COLD=0; done
fi
[[ $COLD -eq 1 ]] && EXTRA+=(--trainer-arg "stream_cold_start=true")
# Mapper depth prior needs continuous ACE mapping (cold start, or --ace-continuous) and dense depth
if [[ $MAPDEPTH -eq 1 ]]; then
    CONT=$COLD; for a in "${EXTRA[@]}"; do [[ "$a" == stream_ace_mode=continuous ]] && CONT=1; done
    [[ $CONT -eq 1 ]] && EXTRA+=(--stream-depth --trainer-arg "stream_ace_mode=continuous" \
        --trainer-arg "stream_mapper_depth=true" --trainer-arg "stream_mapper_depth_weight=$MAPDEPTH_W")
fi
[[ $LBFGS -eq 1 ]] && EXTRA+=(--trainer-arg "lbfgs_refine=true" --trainer-arg "lbfgs_every=$LBFGS_EVERY")
# 3DGUT by default; an explicit --refine-poses wins over the default (UT has no pose gradients)
[[ "$UT" == default ]] && { [[ $REFINE_POSES -eq 1 ]] && UT=0 || UT=1; }
[[ $UT -eq 1 ]] && EXTRA+=(--with_ut)
if [[ $DIFIX -eq 1 ]]; then
    EXTRA+=(--difix-online --trainer-arg "difix_cycle_start_frac=$DIFIX_START")
    [[ "$DIFIX_SHARE" == none ]] || EXTRA+=(--trainer-arg "difix_pseudo_share=$DIFIX_SHARE")
fi

[[ "$CAPTURE_FPS" =~ ^[0-9]+$ && "$CAPTURE_FPS" -gt 0 ]] \
    || { echo "[stream-demo] ERROR: --fps must be a positive integer" >&2; exit 2; }
EXTRA+=(--trainer-arg "stream_capture_fps=$CAPTURE_FPS")

# Latent (JEPA-style) loss: off unless --jepa. Tuning flags without --jepa are ignored, with a warning.
if [[ $JEPA -eq 1 ]]; then
    [[ "$JEPA_BACKEND" == depth_anything || "$JEPA_BACKEND" == dinov2 ]] \
        || { echo "[stream-demo] ERROR: --jepa-backend must be depth_anything or dinov2" >&2; exit 2; }
    if [[ "$JEPA_BACKEND" == depth_anything && "$JEPA_LAYER" != -1 && ! "$JEPA_LAYER" =~ ^(3|6|9|12)$ ]]; then
        echo "[stream-demo] ERROR: --jepa-layer for depth_anything must be -1 or 3, 6, 9, 12" >&2; exit 2
    fi
    EXTRA+=(--trainer-arg "latent_loss=true" --trainer-arg "latent_lambda=$JEPA_LAMBDA"
            --trainer-arg "latent_warmup_steps=$JEPA_WARMUP" --trainer-arg "latent_anneal_end=$JEPA_ANNEAL"
            --trainer-arg "latent_backend=$JEPA_BACKEND" --trainer-arg "latent_layer=$JEPA_LAYER"
            --trainer-arg "latent_side=$JEPA_SIDE")
elif [[ $JEPA_TUNED -eq 1 ]]; then
    echo "[stream-demo] WARNING: --jepa-* tuning flags ignored without --jepa" >&2
fi

fail() { echo "[stream-demo] ERROR: $*" >&2; exit 1; }
warn() { echo "[stream-demo] WARNING: $*" >&2; }
info() { echo "[stream-demo] $*"; }

[[ $UT -eq 1 && $REFINE_POSES -eq 1 ]] && warn "--ut has no pose gradients: --refine-poses keeps only exposure"
[[ $UT -eq 0 && $REFINE_POSES -eq 1 ]] && info "--refine-poses: using the standard rasterizer (3DGUT has no pose gradients)"

# ---------------------------------------------------------------- environments
PY="$ROOT/.venv_3dgrut/bin/python"
ACE_PY="${HYPERSPLAT_ACEZERO_PYTHON:-$ROOT/../.conda_envs/ace0/bin/python}"
[[ -x "$PY" ]] || fail "3DGS env not found at .venv_3dgrut"
[[ -x "$ACE_PY" ]] || fail "ACE-Zero env not found at $ACE_PY (set HYPERSPLAT_ACEZERO_PYTHON)"
"$PY" -c "import pyarrow.flight" 2>/dev/null || fail "pyarrow missing in .venv_3dgrut (pip install 'pyarrow>=17')"
LD_PRELOAD=/usr/lib64/libstdc++.so.6 "$ACE_PY" -c "import torch, pyarrow, dsacstar" 2>/dev/null \
    || fail "ACE env cannot import pyarrow/dsacstar (pip install pyarrow==17.0.0 in ace0)"

# ---------------------------------------------------------------- phone
if [[ $PHONE -eq 1 ]]; then
    [[ "$PHONE_FACING" == back || "$PHONE_FACING" == front ]] || fail "--phone-camera must be back or front"
    det=$("$PY" -m hypersplat.stream.phone detect) || fail "phone detection failed"
    IFS=$'\t' read -r PMODE PDEV PSERIAL PLABEL < <("$PY" -c '
import json, sys
d = json.load(sys.stdin)
print(d["mode"] or "-", d["device"] or "-", d["serial"] or "-", d["label"] or "-", sep="\t")' <<<"$det")
    case "$PMODE" in
        uvc)
            info "phone  : $PLABEL on $PDEV"
            [[ "$PHONE_FACING" == front ]] && warn "in webcam mode the lens is chosen on the phone (use its flip button)"
            SOURCE="$PDEV" ;;
        android)
            # The trainer starts the camera stream over adb and stops it when it exits
            info "phone  : $PLABEL, $PHONE_FACING camera"
            SOURCE="android://$PSERIAL?camera=$PHONE_FACING&fps=$CAPTURE_FPS" ;;
        *)
            echo "[stream-demo] No usable Android camera found:" >&2
            "$PY" -c 'import json, sys; [print("  -", p) for p in json.load(sys.stdin)["problems"]]' <<<"$det" >&2
            cat >&2 <<'HELP'
  On the phone:
    1. Settings > About phone > Software version: tap 7 times (Developer options appear)
    2. Settings > System management / Additional settings > Developer options > USB debugging: on
    3. USB mode: File transfer; unlock the phone and tap "Allow" on "Allow USB debugging?"
  (Android 14+ phones that still offer "Webcam" in the USB menu work without these steps.)
HELP
            exit 1 ;;
    esac
fi

# ---------------------------------------------------------------- wifi camera
if [[ $WIFI -eq 1 ]]; then
    info "wifi   : scanning the local network for cameras..."
    mapfile -t WCAMS < <("$PY" -m hypersplat.stream.wifi | "$PY" -c '
import json, sys
for c in json.load(sys.stdin): print(c["url"])')
    if [[ ${#WCAMS[@]} -eq 0 ]]; then
        fail "no camera found on the network. On the phone: join the same Wi-Fi as this PC (mobile data off), start the camera app's server, then retry; or pass its URL directly"
    elif [[ ${#WCAMS[@]} -eq 1 ]]; then
        SOURCE="${WCAMS[0]%% *}"
    else
        echo "[stream-demo] Cameras found:" >&2
        for i in "${!WCAMS[@]}"; do echo "  $((i+1))) ${WCAMS[$i]}" >&2; done
        read -r -p "[stream-demo] Pick one [1]: " pick </dev/tty
        SOURCE="${WCAMS[$(( ${pick:-1} - 1 ))]%% *}"
    fi
    [[ -n "$SOURCE" ]] || fail "invalid choice"
    info "wifi   : using $SOURCE"
fi

# ---------------------------------------------------------------- usb camera
# Default source: an external USB camera, else an Android phone on the USB cable (webcam mode
# or USB debugging). The laptop's built-in camera is only used when asked for (--builtin).
usb_detect() { "$PY" -m hypersplat.stream.phone usb "$PHONE_FACING" || fail "USB camera detection failed"; }
usb_field() { "$PY" -c 'import json, sys; d = json.load(sys.stdin); v = d[sys.argv[1]]
print("\n".join(v) if isinstance(v, list) else v or "")' "$1"; }
if [[ $PHONE -eq 0 && $WIFI -eq 0 && "$SOURCE" == usb ]]; then
    det=$(usb_detect)
    SOURCE=$(usb_field device <<<"$det")
    if [[ -z "$SOURCE" ]]; then
        echo "[stream-demo] No external USB camera or phone found. Cameras present:" >&2
        usb_field cameras <<<"$det" | sed 's/^/  - /' >&2
        if [[ -t 0 ]]; then
            echo "[stream-demo] >>> Connect an external USB camera (or a phone with USB debugging) now." >&2
            echo "[stream-demo]     Waiting... Ctrl+C to cancel, or rerun with --builtin for the laptop camera" >&2
            last=""
            until [[ -n "$SOURCE" ]]; do
                sleep 1
                det=$(usb_detect)
                SOURCE=$(usb_field device <<<"$det")
                problem=$(usb_field problems <<<"$det")
                if [[ -n "$problem" && "$problem" != "$last" ]]; then
                    echo "[stream-demo]     $problem" >&2
                fi
                last="$problem"
            done
            info "camera connected: $(usb_field label <<<"$det") ($SOURCE)"
            sleep 2  # let the driver finish setting up the device before it is opened
        else
            usb_field problems <<<"$det" | sed 's/^/  /' >&2
            echo "  Plug in a USB camera (check: ls /dev/video*) or a phone with USB debugging," >&2
            echo "  or pick one with --camera N / --builtin." >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------- source
LIVE=1
case "$SOURCE" in
    [0-9]|[0-9][0-9]|/dev/video*)
        dev="$SOURCE"; [[ "$dev" == /dev/* ]] || dev="/dev/video$SOURCE"
        [[ -e "$dev" ]] || fail "no camera at $dev (cameras: $(ls /dev/video* 2>/dev/null | tr '\n' ' '))"
        # Open it briefly: catches a busy/unsupported device before ACE and training spin up
        probe=$("$PY" - "$SOURCE" <<'EOF' 2>/dev/null || true
import sys, cv2
src = sys.argv[1]
cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
ok, frame = cap.read() if cap.isOpened() else (False, None)
print(f"{frame.shape[1]}x{frame.shape[0]} @ {cap.get(cv2.CAP_PROP_FPS) or 30:.0f} fps" if ok else "")
cap.release()
EOF
)
        [[ -n "$probe" ]] || fail "camera $dev could not be read (in use by another app?)"
        name=$(cat "/sys/class/video4linux/$(basename "$dev")/name" 2>/dev/null || echo camera)
        info "camera : $dev ($name, $probe)" ;;
    rtsp://*|rtmp://*|http://*|https://*|udp://*|tcp://*|flight://*|grpc://*|android://*) ;;
    *)  [[ -f "$SOURCE" ]] || fail "source not found: $SOURCE"
        LIVE=0 ;;
esac

# ---------------------------------------------------------------- resources
# Model weights on the data drive, never ~/.cache on the (small) root disk
HF_CACHE="$ROOT/../hf_cache"
[[ -d "$HF_CACHE" ]] && export HF_HOME="${HF_HOME:-$(cd "$HF_CACHE" && pwd)}"
if [[ $DIFIX -eq 1 && ! -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--difix_ref/snapshots" ]]; then
    warn "Difix weights (~5 GB) are not cached yet and will be downloaded to ${HF_HOME:-$HOME/.cache/huggingface}"
fi
mkdir -p "$OUTPUT"
avail_gb=$(df -Pk "$OUTPUT" | awk 'NR==2 {print int($4 / 1048576)}')
(( avail_gb >= 2 )) || fail "only ${avail_gb} GB free for $OUTPUT"
shm_mb=$(df -Pk /dev/shm | awk 'NR==2 {print int($4 / 1024)}')
(( shm_mb >= 512 )) || warn "/dev/shm has only ${shm_mb} MB free (ACE frames + map live there)"
if command -v nvidia-smi >/dev/null; then
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    (( free_mb >= 3500 )) || warn "only ${free_mb} MB GPU memory free; close other viewers/players for full-resolution Difix"
fi

# ---------------------------------------------------------------- CUDA MPS
# A private MPS daemon (its own pipe dir, so other GPU jobs are unaffected): the trainer and
# the ACE sidecar become clients of one GPU context and run kernels concurrently; each gets an
# SM cap, and the sidecar optionally a memory cap (HYPERSPLAT_SIDECAR_MPS_* -> AceSidecar)
MPS_DIR=""
mps_stop() {
    [[ -n "$MPS_DIR" ]] || return 0
    CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" nvidia-cuda-mps-control <<<quit >/dev/null 2>&1 || true
    rm -rf "$MPS_DIR"
    MPS_DIR=""
}
if [[ -n "${MPS:-}" ]]; then
    command -v nvidia-cuda-mps-control >/dev/null || fail "--mps: nvidia-cuda-mps-control not found"
    IFS=, read -r MPS_T MPS_S MPS_MEM <<<"$MPS"
    MPS_DIR="/dev/shm/hypersplat-mps-$$"
    mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
    if nvidia-cuda-mps-control -d; then
        trap mps_stop EXIT
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${MPS_T:-100}"
        export HYPERSPLAT_SIDECAR_MPS_PCT="${MPS_S:-100}"
        [[ -n "${MPS_MEM:-}" ]] && export HYPERSPLAT_SIDECAR_MPS_MEM="0=$MPS_MEM"
        info "MPS    : on (trainer ${MPS_T:-100}% SMs, ACE sidecar ${MPS_S:-100}%${MPS_MEM:+, sidecar memory cap $MPS_MEM})"
    else
        warn "--mps: the MPS daemon did not start; continuing time-sliced"
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        rm -rf "$MPS_DIR"; MPS_DIR=""
    fi
fi

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [[ $VIEW -eq 1 ]]; then
    port_free "$VIEWER_PORT" || fail "viewer port $VIEWER_PORT is in use (--viewer-port P)"
    port_free "$DASH_PORT" || fail "dashboard port $DASH_PORT is in use (--dashboard-port P)"
    EXTRA+=(--stream-viewer-port "$VIEWER_PORT" --stream-dashboard-port "$DASH_PORT"
            --stream-dashboard-host "$DASH_HOST")
    [[ "$DASH_HOST" == 0.0.0.0 ]] && warn "--share: the camera feed is reachable from the network"
    "$PY" -c 'from hypersplat.stream.walk_bounds import walk_client_installed as w; raise SystemExit(not w())' \
        || warn "viewer walk mode unavailable: run scripts/viewer/build_walk_client.sh (free-fly only)"
fi

STOP_FILE="$OUTPUT/results/stream_3dgs/STOP"
REC_FILE="$OUTPUT/results/stream_3dgs/STOP_RECORDING"
rm -f "$STOP_FILE" "$REC_FILE"

info "source : $SOURCE ($([[ $LIVE -eq 1 ]] && echo live || echo video file))"
info "output : $OUTPUT/results/stream_3dgs"
# Effective configuration after defaults: the first log is enough to see what actually ran
info "config : cold-start=$COLD (ace $([[ $COLD -eq 1 ]] && echo continuous || echo refit))  mapper-depth=$MAPDEPTH (w=$MAPDEPTH_W)  lbfgs=$LBFGS (every $LBFGS_EVERY)"
info "         3dgut=$UT  difix=$DIFIX (share $DIFIX_SHARE, start $DIFIX_START)  seed=$SEED  mps=${MPS:-off}  steps=${EXTRA_STEPS:-auto}"
if [[ $JEPA -eq 1 ]]; then
    info "         jepa=on ($JEPA_BACKEND layer $JEPA_LAYER side $JEPA_SIDE, lambda $JEPA_LAMBDA, warmup $JEPA_WARMUP, anneal-end $JEPA_ANNEAL)"
else
    info "         jepa=off"
fi
info "command: PYTHONPATH=$ROOT $PY -u scripts/pipeline/automated_intelligent_pipeline.py $SOURCE --stream --output_dir $OUTPUT ${EXTRA[*]}"
info "options: ${EXTRA[*]}"
if [[ $PHONE -eq 1 || "$SOURCE" == android://* ]]; then
    info "tip    : walk slowly around the subject; stay on the 1x lens, no zoom, stabilization off,"
    info "         focus/exposure locked if the camera app allows it (one fixed camera per stream)"
elif [[ $LIVE -eq 1 && "$SOURCE" != *://* ]]; then
    info "tip    : move the camera (or the object) slowly; the first map needs 24 keyframes with parallax"
fi
if [[ "$SOURCE" == flight://* || "$SOURCE" == grpc://* ]]; then
    info "push a camera from another machine with:"
    info "    python -m hypersplat.stream.flight push 0 grpc://$(hostname -I 2>/dev/null | awk '{print $1}'):${SOURCE##*:}"
fi

# ---------------------------------------------------------------- run
# Own process group: Ctrl+C reaches only this script, which asks the trainer to finish
# (STOP file) instead of killing it mid-step.
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" setsid "$PY" -u scripts/pipeline/automated_intelligent_pipeline.py "$SOURCE" \
    --stream --output_dir "$OUTPUT" "${EXTRA[@]}" &
PID=$!

if [[ $VIEW -eq 1 ]]; then
    DASH_URL="http://localhost:$DASH_PORT"
    (   # announce / open the split view as soon as the dashboard answers
        until (exec 3<>"/dev/tcp/127.0.0.1/$DASH_PORT") 2>/dev/null; do
            kill -0 "$PID" 2>/dev/null || exit 0
            sleep 1
        done
        info "split view (3D viewer | camera): $DASH_URL"
        if [[ $BROWSER -eq 1 && -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
            xdg-open "$DASH_URL" >/dev/null 2>&1 || true
        fi
    ) &
    OPENER=$!
fi

STOPPING=0
on_interrupt() {
    if [[ $STOPPING -eq 0 ]]; then
        STOPPING=1
        mkdir -p "$(dirname "$STOP_FILE")"
        touch "$STOP_FILE"
        echo -e "\n[stream-demo] Stopping gracefully (final checkpoint + PLY). Ctrl+C again to kill."
    else
        echo -e "\n[stream-demo] Killing."
        kill -TERM -- "-$PID" 2>/dev/null || true
    fi
}
trap on_interrupt INT TERM

# Live source in a terminal: "s" stops recording (the trainer sees STOP_RECORDING, releases
# the camera and finishes on the keyframes collected, so frames do not pile up forever)
KEYS=0
if [[ $LIVE -eq 1 && -t 0 ]]; then
    KEYS=1
    info "press s to stop recording (camera off, training finishes on what was captured); Ctrl+C to end now"
fi
status=0 REAPED=0
while kill -0 "$PID" 2>/dev/null; do
    if [[ $KEYS -eq 1 ]]; then
        if read -rsn1 -t 1 key && [[ "$key" == [sS] ]]; then
            mkdir -p "$(dirname "$REC_FILE")"
            touch "$REC_FILE"
            KEYS=0
            info "Recording stopped; training continues on the keyframes collected (Ctrl+C to end now)"
        fi
    else
        wait "$PID" && status=0 || status=$?
        REAPED=1
    fi
done
[[ $REAPED -eq 1 ]] || { wait "$PID" && status=0 || status=$?; }
trap - INT TERM
[[ -n "${OPENER:-}" ]] && kill "$OPENER" 2>/dev/null || true
rm -f "$STOP_FILE" "$REC_FILE"

RESULT="$OUTPUT/results/stream_3dgs"
PLY=$(ls -t "$RESULT"/ply/*.ply 2>/dev/null | head -1 || true)
CKPT=$(ls -t "$RESULT"/ckpts/*.pt 2>/dev/null | head -1 || true)
if [[ $STOPPING -eq 1 && -z "$CKPT" ]]; then
    info "Stopped before training started; nothing was saved."
    exit 0
elif [[ $status -eq 0 ]]; then
    info "Done. PLY: ${PLY:-none}"
    [[ -n "$CKPT" ]] && info "View it again: $PY examples/simple_viewer.py --ckpt $CKPT"
else
    echo "[stream-demo] Pipeline exited with status $status (log above)." >&2
fi
exit "$status"
