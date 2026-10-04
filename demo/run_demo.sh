#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

source "../.venv_3dgrut/bin/activate"

python demo_server.py
