#!/usr/bin/env python3
"""Small-scale A/B tests of 3DGS training hyperparameters on the stream pipeline.

A spec (JSON) lists captures, the pipeline flags every run shares, and named variants that
override trainer options. The baseline is repeated to measure run-to-run noise (ACE and MCMC
are not deterministic), so a variant only counts as a real difference when it clears that
noise. Runs are sequential (one 6 GB GPU) and resumable: finished runs are read back from
results.jsonl and skipped.

    python scripts/experiments/ab_test.py experiments/ab_3dgs_drone.json            # run
    python scripts/experiments/ab_test.py experiments/ab_3dgs_drone.json --dry-run  # commands only
    python scripts/experiments/ab_test.py experiments/ab_3dgs_drone.json --report   # summary only

Spec keys:
    name, output         experiment name, results directory
    captures             video files (or other stream sources)
    flags                pipeline flags shared by every run (e.g. "--stream-depth")
    repeats              {"baseline": 3, "default": 1}
    variants             {"name": {"trainer_key": value, ...}}; "baseline" must exist ({}).
                         Keys are trainer options (simple_trainer.py Config), e.g.
                         "ssim_lambda", "strategy.cap-max"; passed as --trainer-arg KEY=VALUE
    random               optional {"n": 4, "seed": 0, "space": {"key": [v1, v2, ...]}}:
                         n extra variants sampled from the space
    keep_checkpoints     false (default) deletes ckpts/ after a run is scored (PLY is kept)
    map_cache            true: every run of a capture trains on one cached ACE map
                         (experiments/map_cache/<capture stem>; the first run builds it).
                         Separate bootstraps of one capture differed by 0.55 dB, which hides
                         knob effects; on one map the baseline repeats within ~0.02 dB
"""

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv_3dgrut" / "bin" / "python"
PIPELINE = ROOT / "scripts" / "pipeline" / "automated_intelligent_pipeline.py"
METRICS = ("psnr", "ssim", "lpips")


# ============================================================================ spec
def load_spec(path: Path) -> Dict:
    spec = json.loads(path.read_text())
    spec.setdefault("flags", [])
    spec.setdefault("repeats", {"baseline": 3, "default": 1})
    variants = dict(spec.get("variants", {}))
    if "baseline" not in variants:
        raise SystemExit("spec: variants must include 'baseline' (usually {})")
    rnd = spec.get("random")
    if rnd and rnd.get("n"):
        rng = random.Random(rnd.get("seed", 0))
        for i in range(int(rnd["n"])):
            variants[f"rand_{i:02d}"] = {k: rng.choice(v) for k, v in rnd["space"].items()}
    spec["variants"] = variants
    spec["output"] = str(ROOT / spec.get("output", f"experiments/results/{spec['name']}"))
    return spec


def planned_runs(spec: Dict) -> List[Dict]:
    runs = []
    for cap in spec["captures"]:
        stem = Path(cap).stem
        for name, overrides in spec["variants"].items():
            n = spec["repeats"].get(name, spec["repeats"].get("default", 1))
            for r in range(int(n)):
                runs.append({"id": f"{stem}__{name}__r{r}", "capture": cap, "variant": name,
                             "repeat": r, "overrides": overrides})
    return runs


def command(spec: Dict, run: Dict, out_dir: Path) -> List[str]:
    cmd = [str(PY), "-u", str(PIPELINE), str(ROOT / run["capture"]) if not Path(run["capture"]).is_absolute()
           else run["capture"], "--stream", "--output_dir", str(out_dir), *spec["flags"]]
    if spec.get("map_cache"):
        cmd += ["--stream-map-cache", str(ROOT / "experiments" / "map_cache" / Path(run["capture"]).stem)]
    for k, v in run["overrides"].items():
        cmd += ["--trainer-arg", f"{k}={v}"]
    return cmd


# ============================================================================ scoring
def score(out_dir: Path, log_text: str) -> Dict:
    """Metrics of a finished run from its stats files and log."""
    res = out_dir / "results" / "stream_3dgs"
    vals = sorted((res / "stats").glob("val_step*.json"),
                  key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    m: Dict = {}
    if vals:
        v = json.loads(vals[-1].read_text())
        m.update({k: v.get(k) for k in METRICS}, num_gs=v.get("num_GS"),
                 final_step=int(re.search(r"(\d+)", vals[-1].stem).group(1)))
    log = log_text.replace("\r", "\n")
    fin = re.findall(r"\[Stream\] Final: (\{.*\})", log)
    if fin:
        try:
            st = ast.literal_eval(fin[-1])
            m.update({k: st.get(k) for k in ("keyframes", "registered", "bootstrap_s", "refits", "pseudo")})
        except (ValueError, SyntaxError):
            pass
    ws = re.findall(r"\[WorldStop\] World reconstructed at step (\d+)", log)
    m["world_stop_step"] = int(ws[-1]) if ws else None
    st = re.findall(r"\[Stream\] step (\d+): coverage settled .*?annealing to step (\d+)", log)
    m["settle_step"], m["settle_end"] = (int(st[-1][0]), int(st[-1][1])) if st else (None, None)
    m["difix_rounds"] = len(re.findall(r"\[Difix-online\] Round \d+ @", log))
    return m


# ============================================================================ run
def _descendants(pid: int) -> List[int]:
    """All processes below pid (via /proc), whatever session they moved to."""
    children: Dict[int, List[int]] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(d.name))
    out, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def _kill_group(proc: subprocess.Popen):
    """Stop a run and everything it started. The ACE sidecar runs in its own session (so its
    own shutdown can sweep an ACE job); a killed trainer never gets to close it, so every
    descendant's process group is signalled, not just the run's."""
    import signal
    groups = {proc.pid}
    tree = _descendants(proc.pid)
    for pid in tree:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        for g in groups:
            try:
                os.killpg(g, sig)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGKILL or not any(_alive_group(g) for g in groups):
            break
    # A killed trainer leaves its tmpfs work dir (ACE frames + map) behind
    for pid in tree:
        shutil.rmtree(f"/dev/shm/hypersplat-stream-{pid}", ignore_errors=True)


def _alive_group(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def run_all(spec: Dict, dry: bool, only: Optional[str]):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {json.loads(l)["id"] for l in results.read_text().splitlines() if l.strip()} if results.exists() else set()
    env = dict(os.environ)
    hf = ROOT.parent / "hf_cache"
    if hf.is_dir():
        env.setdefault("HF_HOME", str(hf))
    todo = [r for r in planned_runs(spec) if r["id"] not in done and (only is None or r["variant"] == only)]
    # Interleave variants (baseline r0, variant A, ..., baseline r1, ...) so drift over the
    # session (thermals, other load) does not line up with one variant
    todo.sort(key=lambda r: (r["repeat"], r["capture"]))
    print(f"[ab] {spec['name']}: {len(todo)} runs to go ({len(done)} done) -> {out}")
    for i, run in enumerate(todo, 1):
        run_dir = out / "runs" / run["id"]
        cmd = command(spec, run, run_dir)
        print(f"[ab] ({i}/{len(todo)}) {run['id']}  {run['overrides'] or '(baseline)'}", flush=True)
        if dry:
            print("     " + " ".join(cmd))
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)  # an interrupted attempt
        run_dir.mkdir(parents=True)
        t0 = time.time()
        with open(run_dir / "run.log", "w") as log:
            # Own process group: stopping the harness takes the pipeline, trainer, ACE sidecar
            # and any ACE refit with it (none left holding the GPU)
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                    start_new_session=True)
            try:
                code = proc.wait()
            except BaseException:
                _kill_group(proc)
                raise
        rec = {**run, "exit": code, "wall_s": round(time.time() - t0, 1)}
        rec.update(score(run_dir, (run_dir / "run.log").read_text(errors="replace")))
        if not spec.get("keep_checkpoints", False):
            shutil.rmtree(run_dir / "results" / "stream_3dgs" / "ckpts", ignore_errors=True)
        with open(results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"     exit {code}, {rec['wall_s']:.0f} s: PSNR {rec.get('psnr')}, SSIM {rec.get('ssim')}, "
              f"LPIPS {rec.get('lpips')}, step {rec.get('final_step')}", flush=True)
    report(spec)


# ============================================================================ report
def _stats(xs: List[float]) -> tuple:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None), len(xs)


def report(spec: Dict) -> str:
    out = Path(spec["output"])
    path = out / "results.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    lines = [f"# A/B: {spec['name']}", "",
             f"Flags: `{' '.join(spec['flags'])}`. Baseline repeats measure run-to-run noise; a variant",
             "is **beyond noise** when its mean differs from the baseline mean by more than",
             "max(2 x baseline std, 0.2 dB PSNR / 0.01 SSIM / 0.01 LPIPS).", ""]
    for cap in spec["captures"]:
        stem = Path(cap).stem
        rows = [r for r in recs if r["capture"] == cap and r.get("exit") == 0 and r.get("psnr") is not None]
        failed = [r["id"] for r in recs if r["capture"] == cap and (r.get("exit") != 0 or r.get("psnr") is None)]
        base = {k: _stats([r[k] for r in rows if r["variant"] == "baseline"]) for k in METRICS}
        floor = {"psnr": 0.2, "ssim": 0.01, "lpips": 0.01}
        lines += [f"## {stem}", "",
                  "| variant | n | PSNR | SSIM | LPIPS | ΔPSNR | ΔLPIPS | steps | wall s | overrides |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        best = None
        for name in spec["variants"]:
            vr = [r for r in rows if r["variant"] == name]
            if not vr:
                continue
            st = {k: _stats([r[k] for r in vr]) for k in METRICS}
            cells, deltas = [], {}
            for k in METRICS:
                mean, sd, _ = st[k]
                cells.append(f"{mean:.3f}" + (f" ± {sd:.3f}" if sd is not None else ""))
                bm, bsd, _ = base[k]
                if name != "baseline" and bm is not None:
                    d = mean - bm
                    thr = max(2 * bsd, floor[k]) if bsd is not None else floor[k]
                    deltas[k] = (d, abs(d) > thr)
            fmt = lambda k: ("" if k not in deltas else f"{deltas[k][0]:+.3f}" + (" **✓**" if deltas[k][1] else ""))
            steps = _stats([r.get("final_step") for r in vr])[0]
            wall = _stats([r.get("wall_s") for r in vr])[0]
            lines.append(f"| {name} | {len(vr)} | {' | '.join(cells)} | {fmt('psnr')} | {fmt('lpips')} | "
                         f"{steps:.0f} | {wall:.0f} | `{json.dumps(spec['variants'][name])}` |")
            if name != "baseline" and "psnr" in deltas and (best is None or deltas["psnr"][0] > best[1]):
                best = (name, deltas["psnr"][0], deltas["psnr"][1])
        bm, bsd, bn = base["psnr"]
        lines.append("")
        if bm is not None:
            lines.append(f"Baseline noise: PSNR std {bsd:.3f} dB over {bn} runs." if bsd is not None
                         else f"Baseline: 1 run only; noise unknown (repeat the baseline).")
        if best:
            lines.append(f"Best variant by PSNR: **{best[0]}** ({best[1]:+.3f} dB, "
                         f"{'beyond' if best[2] else 'within'} noise).")
        if failed:
            lines.append(f"Failed runs: {', '.join(failed)}")
        lines.append("")
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text)
    print(text)
    return text


def main():
    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--report", action="store_true", help="only (re)write the summary")
    ap.add_argument("--only", help="run only this variant")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    if a.report:
        report(spec)
    else:
        run_all(spec, a.dry_run, a.only)


if __name__ == "__main__":
    main()
