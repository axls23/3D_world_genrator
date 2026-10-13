"""Summaries of a torch.profiler window for the streaming pipeline's CUDA work.

Parses the exported Chrome trace (stable across torch 2.0 and 2.6, so the ACE sidecar can use
it too) into what the GPU-sharing work needs: step times, GPU busy/idle time, host syncs,
device-to-host copies, busy time per stream and the heaviest kernels. Python 3.8-compatible.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

SYNC_CALLS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def _union_ms(intervals: List[Tuple[float, float]]) -> float:
    """Total length of the union of [start, end) intervals, us -> ms."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1000.0


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarize_trace(trace_path: str, steps: Sequence[Tuple[int, float]]) -> Dict:
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    gpu, per_stream = [], defaultdict(list)
    kernels = defaultdict(lambda: [0, 0.0])
    syncs = defaultdict(lambda: [0, 0.0])
    memcpy = defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat, name, ts, dur = e.get("cat", ""), e.get("name", ""), float(e["ts"]), float(e["dur"])
        if cat in GPU_CATS:
            gpu.append((ts, ts + dur))
            per_stream[str((e.get("args") or {}).get("stream", e.get("tid")))].append((ts, ts + dur))
            if cat == "kernel":
                k = kernels[name[:120]]
                k[0] += 1
                k[1] += dur / 1000.0
            else:
                m = memcpy[name]
                m[0] += 1
                m[1] += dur / 1000.0
        elif cat == "cuda_runtime" and any(s in name for s in SYNC_CALLS):
            sc = syncs[name]
            sc[0] += 1
            sc[1] += dur / 1000.0

    step_ms = [(t1 - t0) * 1000.0 for (_, t0), (_, t1) in zip(steps, steps[1:])]
    window_ms = (steps[-1][1] - steps[0][1]) * 1000.0 if len(steps) > 1 else 0.0
    busy = _union_ms(gpu)
    top = sorted(kernels.items(), key=lambda kv: -kv[1][1])[:20]
    return {
        "steps": max(len(steps) - 1, 0),
        "window_ms": round(window_ms, 1),
        "step_ms": {"p50": round(_pct(step_ms, 0.5), 2), "p95": round(_pct(step_ms, 0.95), 2),
                    "max": round(max(step_ms), 2) if step_ms else None},
        "gpu_busy_ms": round(busy, 1),
        # Trace clock and wall clock differ in origin, not in rate: busy / window is the share
        "gpu_busy_frac": round(busy / window_ms, 3) if window_ms else None,
        "per_stream_busy_ms": {k: round(_union_ms(v), 1) for k, v in sorted(per_stream.items())},
        "host_syncs": {k: {"n": v[0], "ms": round(v[1], 1)} for k, v in sorted(syncs.items())},
        "host_syncs_per_step": round(sum(v[0] for v in syncs.values()) / max(len(step_ms), 1), 2),
        "memcpy": {k: {"n": v[0], "ms": round(v[1], 2)} for k, v in sorted(memcpy.items())},
        "top_kernels": [{"name": k, "n": v[0], "ms": round(v[1], 1)} for k, v in top],
    }


def write_profile(prof, steps: Sequence[Tuple[int, float]], out_dir: str, tag: str) -> str:
    """Export the trace, an op table and the summary JSON; returns the summary path."""
    os.makedirs(out_dir, exist_ok=True)
    trace = os.path.join(out_dir, "trace_%s.json" % tag)
    prof.export_chrome_trace(trace)
    sort = "self_cuda_time_total"
    try:
        table = prof.key_averages().table(sort_by=sort, row_limit=40)
    except (KeyError, AttributeError, RuntimeError):
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    with open(os.path.join(out_dir, "ops_%s.txt" % tag), "w") as f:
        f.write(table)
    summary = summarize_trace(trace, steps)
    summary["tag"] = tag
    summary["first_step"] = steps[0][0] if steps else None
    out = os.path.join(out_dir, "summary_%s.json" % tag)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    return out
