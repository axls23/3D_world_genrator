#!/usr/bin/env bash
# The container variant of ops/up.sh: build the image and run the data layers as one podman pod.
#
#   ops/pod.sh up       build the image, start the pod (needs models/: run train_models first)
#   ops/pod.sh status   pod/container state + port checks
#   ops/pod.sh down     stop and remove the pod (the named volumes, so history, are kept)
#   ops/pod.sh reset    down and delete the volumes: the next up starts from zero (PDF row 21)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
POD=hypersplat-streaming
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

up() {
    [[ -d "$ROOT/hypersplat/streaming/models" ]] || { echo "no models/: run python -m hypersplat.streaming.train_models first" >&2; exit 1; }
    podman build -q -t hypersplat-streaming -f "$ROOT/hypersplat/streaming/Containerfile" "$ROOT"
    podman kube play --replace "$ROOT/hypersplat/streaming/k8s/pod.yaml"
    for _ in $(seq 1 60); do port_open 9094 && port_open 5433 && port_open 8501 && break; sleep 2; done
    status
}
status() {
    podman pod ps --filter "name=$POD" --format '{{.Name}} {{.Status}}'
    podman ps --pod --filter "pod=$POD" --format '  {{.Names}} {{.Status}}'
    for p in 9094:kafka 5433:db 8501:dashboard; do
        port_open "${p%%:*}" && echo "${p##*:}  localhost:${p%%:*} open" || echo "${p##*:}  localhost:${p%%:*} CLOSED"
    done
}
down() { podman kube down "$ROOT/hypersplat/streaming/k8s/pod.yaml" 2>&1 | tail -2 || true; }
reset() { down; podman volume rm -f hypersplat-kafka-data hypersplat-db-data >/dev/null; echo "volumes removed"; }

case "${1:-}" in up) up ;; status) status ;; down) down ;; reset) reset ;; *) sed -n 2,8p "$0"; exit 2 ;; esac
