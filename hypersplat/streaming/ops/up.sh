#!/usr/bin/env bash
# Start the streaming stack in the fixed order of Policy & Process Map row 19:
#   1 broker (+ database)  2 topics  3 anomaly models  4 processor  5 dashboard  6 producer
# Each layer only starts once the one before it is up, so no consumer meets a missing topic or
# model. The producer is left to the user: step 6 prints the replay / live command.
#
#   ops/up.sh [up] [--mode auto|compose|local] [--retrain] [--dry-run]
#   ops/up.sh down [--all] [--mode ...] [--dry-run]
#       stop processor + dashboard; Kafka and PostgreSQL keep running (topics and tables stay
#       reachable for a reset). --all in compose mode also runs "compose down" (volumes kept).
#       ops/infra.sh containers are never stopped here: other users may share them.
#
# compose: hypersplat/streaming/docker-compose.yml (Kafka, kafka-init, db, processor, dashboard).
# local:   ops/infra.sh up for Kafka + PostgreSQL, processor and dashboard as background
#          processes with pid files and logs in hypersplat/streaming/results/ (git-ignored).
# auto:    compose when a compose CLI exists and the Kafka and DB ports are free or already
#          ours; local otherwise (e.g. ops/infra.sh's hs-kafka / hs-db hold them).
# Model training always runs on the host: HYPERSPLAT_PYTHON (default python3) needs
# requirements.txt installed; extra train_models arguments go in HYPERSPLAT_TRAIN_ARGS.
set -euo pipefail

OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(dirname "$OPS")                    # hypersplat/streaming
ROOT=$(cd "$PKG/../.." && pwd)           # repository root
COMPOSE_FILE="$PKG/docker-compose.yml"
RESULTS="$PKG/results/up"
PY=${HYPERSPLAT_PYTHON:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CMD=up MODE=${HYPERSPLAT_UP_MODE:-auto} DRY=0 RETRAIN=0 ALL=0
while (($#)); do
    case "$1" in
        up|down) CMD=$1 ;;
        --mode) MODE=${2:?--mode needs auto, compose or local}; shift ;;
        --mode=*) MODE=${1#*=} ;;
        --dry-run|-n) DRY=1 ;;
        --retrain) RETRAIN=1 ;;
        --all) ALL=1 ;;
        -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ $MODE == auto || $MODE == compose || $MODE == local ]] || { echo "bad --mode $MODE" >&2; exit 1; }

# Ports and model paths come from the contract (the only place they are defined).
read -r KAFKA_BOOTSTRAP KAFKA_PORT DB_PORT DASHBOARD_PORT MODELS_READY < <("$PY" - <<'EOF'
import os
from hypersplat.streaming import contract as c
ready = all(os.path.exists(c.model_path(n)) for n in (c.POSE_MODEL, c.TRAINING_MODEL))
print(c.KAFKA_BOOTSTRAP, c.KAFKA_EXTERNAL_PORT, c.DB_PORT, c.DASHBOARD_PORT, int(ready))
EOF
)

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
step() { printf '\n[%s/6] %s\n' "$1" "$2"; }
run() {  # print the command; execute it unless --dry-run
    printf '  $ %s\n' "$*"
    ((DRY)) || "$@"
}

COMPOSE=()
if docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE=(docker-compose)
elif podman compose version >/dev/null 2>&1; then COMPOSE=(podman compose)
fi
DC=("${COMPOSE[@]}" -f "$COMPOSE_FILE")

if [[ $MODE == auto ]]; then
    MODE=local
    if ((${#COMPOSE[@]})); then
        MODE=compose
        for svc_port in "kafka:$KAFKA_PORT" "db:$DB_PORT"; do
            svc=${svc_port%%:*} port=${svc_port#*:}
            if port_open "$port" && [[ -z "$("${DC[@]}" ps -q "$svc" 2>/dev/null)" ]]; then
                echo "port $port ($svc) is held outside this compose project; using local mode"
                MODE=local
                break
            fi
        done
    fi
elif [[ $MODE == compose ]] && ! ((${#COMPOSE[@]})); then
    if ((DRY)); then DC=(docker compose -f "$COMPOSE_FILE")  # show the commands anyway
    else echo "no compose CLI found (docker compose / docker-compose / podman compose)" >&2; exit 1
    fi
fi
echo "mode: $MODE$( ((DRY)) && echo ' (dry run)')"

start_bg() {  # name, command...: background process with pid file + log under results/
    local name=$1; shift
    local pidf="$RESULTS/$name.pid" log="$RESULTS/$name.log"
    if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "  $name already running (pid $(cat "$pidf"), log $log)"
        return
    fi
    printf '  $ %s > %s 2>&1 &\n' "$*" "$log"
    ((DRY)) && return
    mkdir -p "$RESULTS"
    (cd "$ROOT" && exec nohup "$@" >"$log" 2>&1) &
    echo $! >"$pidf"
    sleep 3  # catch an immediate crash (missing module, bad DSN) here, not in the dashboard
    if ! kill -0 "$(cat "$pidf")" 2>/dev/null; then
        echo "  $name exited at startup; last lines of $log:" >&2
        tail -n 20 "$log" >&2
        rm -f "$pidf"
        exit 1
    fi
    echo "  $name pid $(cat "$pidf"), log $log"
}

stop_bg() {
    local pidf="$RESULTS/$1.pid"
    [[ -f $pidf ]] || { [[ $MODE == local ]] && echo "  $1: no pid file, not running"; return 0; }
    local pid; pid=$(cat "$pidf")
    if kill -0 "$pid" 2>/dev/null; then
        printf '  $ kill %s   # %s\n' "$pid" "$1"
        ((DRY)) || kill "$pid"
    fi
    ((DRY)) || rm -f "$pidf"
}

if [[ $CMD == down ]]; then
    stop_bg dashboard
    stop_bg processor
    if [[ $MODE == compose ]]; then
        if ((ALL)); then run "${DC[@]}" down   # volumes kept; "down -v" also wipes the data
        else run "${DC[@]}" stop dashboard processor
        fi
    fi
    exit 0
fi

step 1 "broker: Kafka (KRaft) + PostgreSQL"
if [[ $MODE == compose ]]; then
    run "${DC[@]}" up -d kafka db
else
    run bash "$OPS/infra.sh" up
fi

step 2 "topics: create the contract topics (idempotent)"
if [[ $MODE == compose ]]; then
    run "${DC[@]}" run --rm kafka-init   # waits for a healthy broker, fills /venv
else
    run "$PY" -m hypersplat.streaming.topics --bootstrap "$KAFKA_BOOTSTRAP" --wait 120 create
fi

step 3 "anomaly models: IsolationForest per source"
if ((MODELS_READY)) && ! ((RETRAIN)); then
    echo "  models present, skipping training (--retrain to refit)"
else
    # shellcheck disable=SC2086  # HYPERSPLAT_TRAIN_ARGS is a word list on purpose
    run "$PY" -m hypersplat.streaming.train_models ${HYPERSPLAT_TRAIN_ARGS:-}
fi

step 4 "processor: Kafka -> features -> model.score -> PostgreSQL"
if [[ $MODE == compose ]]; then
    run "${DC[@]}" up -d --no-deps processor
else
    start_bg processor "$PY" -m hypersplat.streaming.processor
fi

step 5 "dashboard: Streamlit on port $DASHBOARD_PORT"
if [[ $MODE == compose ]]; then
    run "${DC[@]}" up -d --no-deps dashboard
else
    start_bg dashboard "$PY" -m streamlit run "$PKG/dashboard.py" \
        --server.port "$DASHBOARD_PORT" --server.headless true --browser.gatherUsageStats false
fi

step 6 "producer: start it yourself (the stack is ready)"
cat <<EOF
  replay a recording:  $PY -m hypersplat.streaming.replay --help
  live (trainer/ACE):  HYPERSPLAT_STREAMING=1 HYPERSPLAT_KAFKA=$KAFKA_BOOTSTRAP <your trainer or ACE command>
  dashboard:           http://localhost:$DASHBOARD_PORT
  health check:        $PY -m hypersplat.streaming.health   (run twice; counts must climb)
  topic counts:        $PY -m hypersplat.streaming.topics list
EOF
