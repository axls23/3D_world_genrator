#!/usr/bin/env bash
# Live stream demo with Kafka streaming on (Policy & Process Map rows 2, 3, 4, 7).
#
#   hypersplat/streaming/live.sh [run_stream_demo.sh options...]
#
# Runs demo/run_stream_demo.sh unchanged with:
#   HYPERSPLAT_STREAMING=1       autohook/sitecustomize.py installs hooks.py in every Python process
#   PYTHONPATH=autohook:...      reaches the pipeline, the trainer subprocess and the ACE sidecar
#   HYPERSPLAT_JOB_ID            one job id shared by the pose and training events
#   tb_every=TRAIN_LOG_EVERY     trainer logs train/* (one TrainingSnapshot) every N steps
# run_stream_demo.sh has no --trainer-arg option of its own (it rejects unknown flags), so
# tb_every travels in HYPERSPLAT_STREAMING_TRAINER_ARGS and the hook appends
# "--trainer-arg tb_every=N" to the pipeline's argv, which passes it to the trainer.
# poses_final.txt and the rest of the file hand-off are unchanged (fallback path).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${HYPERSPLAT_PYTHON:-$ROOT/.venv_3dgrut/bin/python}"
[[ -x "$PY" ]] || PY=python3   # only reads contract.py (stdlib); the demo checks its own envs

OUTPUT="demo/output/stream_demo"   # run_stream_demo.sh default; only names the job
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == --output && $((i + 1)) -lt ${#args[@]} ]]; then
        OUTPUT="${args[i + 1]}"
    fi
done

TB_EVERY="$(cd "$ROOT" && "$PY" -c 'from hypersplat.streaming.contract import TRAIN_LOG_EVERY as n; print(n)')"

export HYPERSPLAT_STREAMING=1
export PYTHONPATH="$HERE/autohook${PYTHONPATH:+:$PYTHONPATH}"
export HYPERSPLAT_JOB_ID="${HYPERSPLAT_JOB_ID:-$(basename "$OUTPUT")-$(date +%Y%m%d-%H%M%S)}"
export HYPERSPLAT_STREAMING_TRAINER_ARGS="tb_every=$TB_EVERY${HYPERSPLAT_STREAMING_TRAINER_ARGS:+ $HYPERSPLAT_STREAMING_TRAINER_ARGS}"

# Say plainly what is not up instead of streaming into the void (nothing is started here: the data layers
# are the pod, ops/pod.sh up).
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
KPORT="${HYPERSPLAT_KAFKA:-localhost:9094}"; KPORT="${KPORT##*:}"
port_open "${KPORT//[^0-9]/}" 2>/dev/null || echo "[streaming] WARNING: Kafka is not answering on :${KPORT}; events will be dropped. Start the data layers: hypersplat/streaming/ops/pod.sh up" >&2
port_open "${HYPERSPLAT_DASHBOARD_PORT:-8501}" || echo "[streaming] Streamlit is not running on :${HYPERSPLAT_DASHBOARD_PORT:-8501} (the 8090 button shows it greyed out). Start it with hypersplat/streaming/ops/pod.sh up" >&2
echo "[streaming] job $HYPERSPLAT_JOB_ID -> Kafka ${HYPERSPLAT_KAFKA:-localhost:9094} (tb_every=$TB_EVERY)" >&2
exec "$ROOT/demo/run_stream_demo.sh" "$@"
