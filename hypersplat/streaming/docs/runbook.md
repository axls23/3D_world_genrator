# HyperSplat streaming runbook

Appendix to the Policy & Process Map (rows 5, 6, 13, 19, 20). It covers how to start, check,
reset and stop the streaming add-on that runs beside the 3D reconstruction pipeline. Commands
run from the repository root with the streaming requirements installed
(`pip install -r hypersplat/streaming/requirements.txt`, already in `.venv_3dgrut`). Set
`HYPERSPLAT_PYTHON` to that interpreter when `python3` is a different one.

Ports, topic names and the database DSN are defined in `hypersplat/streaming/contract.py`, and
each one can be overridden with a `HYPERSPLAT_<NAME>` environment variable:

| Layer | Where | Host port | In-network address |
|---|---|---|---|
| Kafka broker (KRaft) | `kafka` service / `hs-kafka` | 9094 (`HYPERSPLAT_KAFKA_PORT`) | `kafka:29092` (`HYPERSPLAT_KAFKA_INTERNAL`) |
| PostgreSQL 16 | `db` service / `hs-db` | 5433 (`HYPERSPLAT_DB_PORT`) | `db:5432` (`HYPERSPLAT_DB_DSN_INTERNAL`) |
| Dashboard (Streamlit) | `dashboard` service | 8501 (`HYPERSPLAT_DASHBOARD_PORT`) | n/a |

Ports 8080, 8081 and 8092 belong to `gsplat-server` in the root `docker-compose.yml`, so the
streaming stack never uses them. `contract.check_ports()` fails at import time if a setting
collides with them.

## 1. Start order

The order is fixed by row 19. Each layer starts only after the one before it is up, so no
consumer starts before its topic or model exists.

| # | Layer | Command (what `ops/up.sh` runs) |
|---|---|---|
| 1 | broker + database | compose: `up -d kafka db`; local: `ops/infra.sh up` |
| 2 | topics | compose: `run --rm kafka-init`; local: `python -m hypersplat.streaming.topics --wait 120 create` |
| 3 | anomaly models | `python -m hypersplat.streaming.train_models` (skipped when both `.joblib` files exist; `--retrain` forces it) |
| 4 | processor | `python -m hypersplat.streaming.processor` |
| 5 | dashboard | `streamlit run hypersplat/streaming/dashboard.py` on port 8501 |
| 6 | producer | started by you, see below |

```bash
bash hypersplat/streaming/ops/up.sh --dry-run   # print the ordered steps only
bash hypersplat/streaming/ops/up.sh             # start everything (auto mode)
```

`--mode auto` is the default. It uses compose when `docker compose`, `docker-compose` or
`podman compose` is installed and the Kafka port is free or already held by this compose
project. Otherwise it falls back to `local`: `ops/infra.sh up` starts the Kafka and PostgreSQL
containers, and the processor and dashboard run as background processes. Their pid files and
logs are in `hypersplat/streaming/results/up/`, which git ignores. Model training always runs on
the host. Extra arguments for `train_models` go in `HYPERSPLAT_TRAIN_ARGS`. In compose mode the
containers read the models from the mounted repository, so keep `HYPERSPLAT_MODEL_DIR` unset
there.

Producer (step 6):

```bash
python -m hypersplat.streaming.replay --help                     # replay a recording
HYPERSPLAT_STREAMING=1 <trainer or ACE command>                   # live hooks
```

Evidence that topics exist and receive messages (row 6). The count is the sum of the topic's
end offsets, and `retained` leaves out segments that have aged out:

```bash
python -m hypersplat.streaming.topics list          # the 3 contract topics
python -m hypersplat.streaming.topics list --all    # every topic on the broker
```

## 2. Health check (row 20)

```bash
python -m hypersplat.streaming.health
```

Run it twice, a few seconds apart, while a producer is running. The pose and snapshot counts
must climb between the two runs, and the anomaly count must be above 0 once injected bad poses
have passed through. If they do not, check each layer in start order:

1. `bash hypersplat/streaming/ops/infra.sh status` (local) or `docker compose -f hypersplat/streaming/docker-compose.yml ps`: Kafka and the database are up and their ports are open.
2. `python -m hypersplat.streaming.topics list`: the message counts climb. If they do not, the producer is not reaching Kafka. See the Windows/WSL section below.
3. Processor log: `hypersplat/streaming/results/up/processor.log`, or `docker compose ... logs processor`.
4. Dashboard: http://localhost:8501. It refreshes every 2 s.

## 3. Reset

This wipes the streamed data and keeps the code and the containers' images.

```bash
bash hypersplat/streaming/ops/up.sh down                      # processor + dashboard only
python -m hypersplat.streaming.topics delete --yes            # drop the 3 topics (waits)
python -m hypersplat.streaming.topics create                  # recreate them empty
psql postgresql://hypersplat:hypersplat@localhost:5433/hypersplat \
     -c 'DROP TABLE IF EXISTS pose_history, training_history, jobs'
rm -f hypersplat/streaming/models/*.joblib                    # only to refit the models
bash hypersplat/streaming/ops/up.sh
```

`up.sh down` leaves Kafka and PostgreSQL running, so the steps after it can still reach them.
Without a host `psql`, run the same statement inside the container:
`podman exec hs-db psql -U hypersplat -c '...'` (local), or
`docker compose -f hypersplat/streaming/docker-compose.yml exec db psql -U hypersplat -c '...'`.

`contract.DDL` only uses `CREATE ... IF NOT EXISTS`, so applying it again on the next start
(the processor's job) recreates the empty tables. A committed consumer
offset that points past the end of a recreated topic falls back to `earliest`.
To keep a past job for comparison, set `archived = true` on its row in `jobs` instead of
dropping it (growth plan, row 15).

With compose, `docker compose -f hypersplat/streaming/docker-compose.yml down -v` wipes
everything in one step, including the Kafka log, the database volume and the `/venv` volume.

## 4. Shutdown

```bash
bash hypersplat/streaming/ops/up.sh down --all  # processor + dashboard; compose: "compose down"
bash hypersplat/streaming/ops/infra.sh down     # local mode only: also stop hs-kafka and hs-db
```

`up.sh down --all` keeps the data: compose volumes stay. In local mode it never touches the
`hs-*` containers, which other users may share, so stopping them is the separate
`infra.sh down` step.
Stop the producers first, since they drop events while the broker is down instead of blocking
the trainer. Their drop counter shows up as `events_dropped` on the dashboard.

## 5. Windows and WSL

The trainer runs on Windows and the ACE pose estimator runs in WSL. Both produce to the same
broker. Kafka runs on Docker Desktop and has two listeners:

- **INTERNAL** `kafka:29092` serves containers on the compose network: kafka-init, the
  processor and the dashboard.
- **EXTERNAL** port 9094 is published on every host interface and advertised as
  `${HYPERSPLAT_KAFKA_ADVERTISED_HOST:-localhost}:9094`. Windows reaches it at
  `localhost:9094`.

From WSL:

- With mirrored networking (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`),
  `localhost:9094` works as it does on Windows, so nothing needs to be set.
- With the default NAT networking, `localhost` in WSL is not the Windows host. Point the
  producer at the Windows host IP, and advertise that IP so the broker's metadata response
  sends clients back to an address they can reach:

  ```bash
  # in WSL
  WIN_IP=$(ip route show default | awk '{print $3}')
  export HYPERSPLAT_KAFKA=$WIN_IP:9094
  ```
  ```powershell
  # on Windows, before starting the stack
  $env:HYPERSPLAT_KAFKA_ADVERTISED_HOST = "<that IP>"
  ```

  With a single advertised host, Windows-side producers then also use that IP, so set
  `HYPERSPLAT_KAFKA=<that IP>:9094` there too, or switch WSL to mirrored networking. If the
  Windows firewall blocks the connection, allow inbound TCP 9094 on the WSL virtual adapter.

`ops/up.sh` is a bash script. On Windows, run it from WSL or Git Bash, or run the compose and
`python -m` commands from the start-order table by hand in PowerShell.
