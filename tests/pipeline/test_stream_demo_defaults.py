"""demo/run_stream_demo.sh default set: CPU-only, no pipeline launch.

The script is the only place the demo's defaults are composed, so these tests pin the default
assignments and the opt-out flags. A silent change to any default fails here.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "demo" / "run_stream_demo.sh"


@pytest.fixture(scope="module")
def text():
    return SCRIPT.read_text()


def test_script_is_valid_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    res = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("assignment", [
    'DIFIX=1 VIEW=1 BROWSER=1',       # online Difix on by default
    'DIFIX_SHARE=0.15 DIFIX_START=0.5 SEED=42',
    'UT=default REFINE_POSES=0',      # 3DGUT on by default (resolved below)
    'COLD=default LBFGS=1 LBFGS_EVERY=50 MAPDEPTH=1 MAPDEPTH_W=50',  # cold start, L-BFGS, mapper depth on
])
def test_default_assignments_unchanged(text, assignment):
    assert assignment in text


def test_ut_resolves_to_on_unless_refine_poses(text):
    assert '[[ "$UT" == default ]] && { [[ $REFINE_POSES -eq 1 ]] && UT=0 || UT=1; }' in text


def test_cold_start_defaults_off_for_bootstrap_maps(text):
    # cold start is disabled when a cached map or a replay needs a bootstrap map
    assert 'stream_replay=*' in text and '--stream-map-cache' in text
    assert re.search(r'COLD=0', text)


def test_opt_out_flags_exist(text):
    for flag in ["--no-difix)", "--no-lbfgs)", "--no-mapper-depth)", "--no-cold-start)", "--no-ut)",
                 "--map-cache)", "--replay)", "--refine-poses)"]:
        assert flag in text, flag


def test_effective_config_banner_present(text):
    assert 'info "config :' in text
    assert 'info "command: PYTHONPATH=$ROOT' in text
    assert "EXTRA_STEPS" in text
