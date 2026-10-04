#!/usr/bin/env bash
# Live streaming 3DGS demo: camera -> keyframes -> ACE sidecar -> live training + online Difix,
# with a split view in the browser: 3DGS viewer (viser) left, camera feed right.
# See docs/getting-started/streaming.md.
#
#   ./run_stream_demo.sh                               # external USB camera (default, auto-detected)
#   ./run_stream_demo.sh --camera 2                    # a specific camera index / /dev/videoN
#   ./run_stream_demo.sh --builtin                     # the laptop's own camera
#   ./run_stream_demo.sh --phone                       # Android phone over USB-C (webcam mode / scrcpy)
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
  --phone-camera C    back (default) or front
  --video [FILE]      use a video file instead of a camera (default demo/input/drone_demo.mp4)
  --realtime          play a video like a live camera (frames drop when busy)
  --record-seconds S  stop recording after S seconds (training continues on what was captured)
  --record-keyframes K
                      stop recording after K keyframes (training continues)
  --steps N           training step ceiling (default 7000 for videos, 10000 for live sources);
                      training stops earlier once the world is reconstructed
  --target-psnr DB    held-out PSNR that counts as reconstructed once gains flatten (default 22)
  --no-auto-stop      train to the step ceiling instead of stopping on convergence
  --depth             dense depth per keyframe (Depth Anything V2, aligned to ACE points):
                      denser seeding + depth loss, ~20 ms per keyframe
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
  --deterministic     reproducible runs of a video file: deterministic kernels for training and
                      ACE maps, file fully localized before step 0 (slower; ~0.2 dB lower)
  --no-view           no viewer / dashboard / browser
  --no-browser        start the dashboard but do not open a browser
  --viewer-port P     viser 3DGS viewer port (default 8080)
  --dashboard-port P  split-view dashboard port (default 8090)
  --share             serve the dashboard on all interfaces (exposes the camera feed!)
  --budget-mb MB      keyframe store memory budget (default 256)
  --output DIR        output directory (default demo/output/stream_demo)
  -h, --help          this help
EOF
}

SOURCE="usb"
OUTPUT="demo/output/stream_demo"
DEMO_VIDEO="demo/input/drone_demo.mp4"
EXTRA=()
DIFIX=1 VIEW=1 BROWSER=1 PHONE=0 PHONE_FACING=back
DIFIX_SHARE=0.15 DIFIX_START=0.5 SEED=42
UT=default REFINE_POSES=0
VIEWER_PORT=8080 DASH_PORT=8090 DASH_HOST=127.0.0.1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera)         SOURCE="$2"; shift ;;
        --builtin)        SOURCE="0" ;;
        --phone)          PHONE=1 ;;
        --phone-camera)   PHONE_FACING="$2"; shift ;;
        --video)          if [[ $# -gt 1 && "$2" != -* ]]; then SOURCE="$2"; shift; else SOURCE="$DEMO_VIDEO"; fi ;;
        --realtime)       EXTRA+=(--stream-pace realtime) ;;
        --steps)          EXTRA+=(--max_steps "$2"); shift ;;
        --target-psnr)    EXTRA+=(--stream-target-psnr "$2"); shift ;;
        --no-auto-stop)   EXTRA+=(--no-early-stopping) ;;
        --record-seconds) EXTRA+=(--stream-record-seconds "$2"); shift ;;
        --record-keyframes) EXTRA+=(--stream-record-keyframes "$2"); shift ;;
        --depth)          EXTRA+=(--stream-depth) ;;
        --refine-poses)   EXTRA+=(--stream-pose-opt --stream-exposure-opt); REFINE_POSES=1 ;;
        --ut)             UT=1 ;;
        --no-ut)          UT=0 ;;
        --map-cache)      EXTRA+=(--stream-map-cache "$2"); shift ;;
        --no-difix)       DIFIX=0 ;;
        --difix-share)    DIFIX_SHARE="$2"; shift ;;
        --difix-start)    DIFIX_START="$2"; shift ;;
        --seed)           SEED="$2"; shift ;;
        --deterministic)  EXTRA+=(--deterministic) ;;
        --no-view)        VIEW=0 ;;
        --no-browser)     BROWSER=0 ;;
        --viewer-port)    VIEWER_PORT="$2"; shift ;;
        --dashboard-port) DASH_PORT="$2"; shift ;;
        --share)          DASH_HOST=0.0.0.0 ;;
        --budget-mb)      EXTRA+=(--stream-budget-mb "$2"); shift ;;
        --output)         OUTPUT="$2"; shift ;;
        -h|--help)        usage; exit 0 ;;
        -*)               echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                SOURCE="$1" ;;
    esac
    shift
done
EXTRA+=(--seed "$SEED")
# 3DGUT by default; an explicit --refine-poses wins over the default (UT has no pose gradients)
[[ "$UT" == default ]] && { [[ $REFINE_POSES -eq 1 ]] && UT=0 || UT=1; }
[[ $UT -eq 1 ]] && EXTRA+=(--with_ut)
if [[ $DIFIX -eq 1 ]]; then
    EXTRA+=(--difix-online --trainer-arg "difix_cycle_start_frac=$DIFIX_START")
    [[ "$DIFIX_SHARE" == none ]] || EXTRA+=(--trainer-arg "difix_pseudo_share=$DIFIX_SHARE")
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
            SOURCE="android://$PSERIAL?camera=$PHONE_FACING" ;;
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

# ---------------------------------------------------------------- usb camera
# Default source: an external USB camera, else an Android phone on the USB cable (webcam mode
# or USB debugging). The laptop's built-in camera is only used when asked for (--builtin).
usb_detect() { "$PY" -m hypersplat.stream.phone usb "$PHONE_FACING" || fail "USB camera detection failed"; }
usb_field() { "$PY" -c 'import json, sys; d = json.load(sys.stdin); v = d[sys.argv[1]]
print("\n".join(v) if isinstance(v, list) else v or "")' "$1"; }
if [[ $PHONE -eq 0 && "$SOURCE" == usb ]]; then
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

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [[ $VIEW -eq 1 ]]; then
    port_free "$VIEWER_PORT" || fail "viewer port $VIEWER_PORT is in use (--viewer-port P)"
    port_free "$DASH_PORT" || fail "dashboard port $DASH_PORT is in use (--dashboard-port P)"
    EXTRA+=(--stream-viewer-port "$VIEWER_PORT" --stream-dashboard-port "$DASH_PORT"
            --stream-dashboard-host "$DASH_HOST")
    [[ "$DASH_HOST" == 0.0.0.0 ]] && warn "--share: the camera feed is reachable from the network"
fi

STOP_FILE="$OUTPUT/results/stream_3dgs/STOP"
REC_FILE="$OUTPUT/results/stream_3dgs/STOP_RECORDING"
rm -f "$STOP_FILE" "$REC_FILE"

info "source : $SOURCE ($([[ $LIVE -eq 1 ]] && echo live || echo video file))"
info "output : $OUTPUT/results/stream_3dgs"
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
setsid "$PY" -u scripts/pipeline/automated_intelligent_pipeline.py "$SOURCE" \
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
