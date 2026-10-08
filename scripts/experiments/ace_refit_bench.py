#!/usr/bin/env python3
"""ACE refit bench: the stream's ACE loop alone (no 3DGS), refits allowed to finish.

Runs a StreamSession on a video until the source is consumed and no refit is pending, then
reports every refit's alignment diagnostics (hypersplat/stream/gauge.py): the similarity the
reconstruction drifted by (scale, rotation, shift) and how far cameras moved after removing it,
relative to the camera path extent, plus registration counts and refit times.

    python scripts/experiments/ace_refit_bench.py demo/input/drone_demo.mp4 \\
        --map-cache experiments/map_cache/drone_demo --recipe ace0 --gauge-lock \\
        --out experiments/refit_bench/ace0_lock.json --lock experiments/.gpu.lock

With a map cache every arm starts from the same bootstrap map and poses.
"""

import argparse
import fcntl
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--map-cache", default=None)
    ap.add_argument("--recipe", choices=["legacy", "ace0"], default="legacy")
    ap.add_argument("--gauge-lock", action="store_true")
    ap.add_argument("--refit-iterations", type=int, default=4000)
    ap.add_argument("--max-keyframes", type=int, default=400)
    ap.add_argument("--extra-refits", type=int, default=0,
                    help="after the stream settles, force this many refits on the same frames: "
                         "with unchanged data any camera motion they cause is loop drift")
    ap.add_argument("--ace-mode", choices=["refit", "continuous"], default="refit")
    ap.add_argument("--tick-hz", type=float, default=10.0,
                    help="continuous mode: simulated 3DGS steps per second (one tick each)")
    ap.add_argument("--run-after-end", type=float, default=600.0,
                    help="continuous mode: keep mapping this long after the source ended")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock", default=None, help="flock this file for the whole run (shared GPU)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from hypersplat.pipeline.params import SceneProfile
    from hypersplat.stream.session import StreamConfig, StreamSession

    lock = open(args.lock, "w") if args.lock else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    cfg = StreamConfig(source=args.video, pace="fast", map_cache=args.map_cache,
                       max_keyframes=args.max_keyframes, refit_iterations=args.refit_iterations,
                       refit_recipe=args.recipe, gauge_lock=args.gauge_lock, ace_mode=args.ace_mode)
    s = StreamSession(cfg, profile=SceneProfile.from_env())
    t0 = time.time()
    s.start()
    try:
        if not s.wait_for_map(timeout=600):
            raise SystemExit(f"no map: {s.failed}")
        pairs_path = Path(args.out).with_suffix(".pairs.npz")
        saved = [0]

        def save_pairs():
            # After every refit, so a stopped run keeps its data for offline re-fits
            if len(s.refit_pairs) != saved[0]:
                pairs_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(pairs_path, **{f"v{v}_{k}": arr for v, (a, b) in dict(s.refit_pairs).items()
                                        for k, arr in (("new", a), ("ref", b))})
                saved[0] = len(s.refit_pairs)

        def settle():
            quiet = None
            while time.time() - t0 < args.timeout:
                time.sleep(2.0)
                save_pairs()
                if s.failed:
                    return
                # finished can flicker True between a map event and the chained refit it starts
                if s.finished:
                    quiet = quiet or time.time()
                    if time.time() - quiet > 10:
                        return
                else:
                    quiet = None

        timeline, first_w2c = [], {}
        if args.ace_mode == "continuous":
            stop_ticks = threading.Event()

            def ticker():
                while not stop_ticks.is_set():
                    s.tick()
                    time.sleep(1.0 / args.tick_hz)

            threading.Thread(target=ticker, daemon=True).start()
            end_seen, t_sample = None, 0.0
            while time.time() - t0 < args.timeout and not s.failed:
                time.sleep(1.0)
                if time.time() - t_sample >= 10:
                    t_sample = time.time()
                    for kf in s.store.registered():
                        first_w2c.setdefault(kf.kid, kf.w2c.copy())
                    st = s.stats()
                    timeline.append({"t": round(t_sample - t0, 1), "registered": st["registered"],
                                     "keyframes": st["keyframes"], "relocs": st.get("relocs", 0),
                                     "ticks": st.get("mapper_ticks", 0), "anchored": st.get("anchored", 0),
                                     "version": st["map_version"]})
                    print(json.dumps(timeline[-1]), flush=True)
                if s.source_ended:
                    end_seen = end_seen or time.time()
                    if time.time() - end_seen > args.run_after_end:
                        break
            stop_ticks.set()
        else:
            settle()
        n_stream = len(s.refit_log)
        for _ in range(args.extra_refits):
            with s._cond:
                s._maybe_refit(force=True)
            settle()
        st = s.stats()
        _, table = s.views()
        cont = None
        if args.ace_mode == "continuous":
            from hypersplat.stream.gauge import fit_sim3
            reg = {kf.kid: kf.w2c for kf in s.store.registered()}
            cen = np.stack([np.linalg.inv(w)[:3, 3] for w in reg.values()]) if reg else np.zeros((1, 3))
            extent = float(np.linalg.norm(cen.max(0) - cen.min(0))) or 1.0
            t_end = time.time()
            windows = []
            for w0 in range(0, int(t_end - t0) + 60, 60):
                mv = np.array([m for (t, _, m) in s.reloc_moves if w0 <= t - t0 < w0 + 60]) / extent
                if len(mv):
                    windows.append({"t0": w0, "relocs": len(mv), "median": float(np.median(mv)),
                                    "p90": float(np.percentile(mv, 90)), "max": float(mv.max()),
                                    "gt1pct": int((mv > 0.01).sum()), "gt5pct": int((mv > 0.05).sum())})
            common = [k for k in reg if k in first_w2c]
            drift = None
            if len(common) >= 3:
                r = fit_sim3(np.stack([reg[k] for k in common]), np.stack([first_w2c[k] for k in common]))
                rel = r.all_res / extent
                drift = {"frames": len(common), "scale": r.sim3.s, "rot_deg": r.sim3.angle_deg,
                         "raw_median": r.raw_center_median / extent, "aligned_median": float(np.median(rel)),
                         "aligned_p90": float(np.percentile(rel, 90)), "gt1pct": int((rel > 0.01).sum()),
                         "gt5pct": int((rel > 0.05).sum())}
            cont = {"timeline": timeline, "reloc_windows": windows, "drift_since_first_registered": drift}
        out = {"recipe": args.recipe, "gauge_lock": args.gauge_lock, "video": args.video,
               "wall_s": round(time.time() - t0, 1), "timed_out": time.time() - t0 >= args.timeout,
               "keyframes": st.get("keyframes"), "registered": st.get("registered"),
               "map_version": st.get("map_version"), "refits": s.refit_log, "stream_refits": n_stream,
               "ace_mode": args.ace_mode, "continuous": cont,
               "stats": {k: v for k, v in st.items() if isinstance(v, (int, float, str, type(None)))}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        # Final poses in the bootstrap frame (gauge-locked) or the last map's frame, by name
        np.savez(Path(args.out).with_suffix(".npz"),
                 names=np.array(table.column("name").to_pylist()),
                 c2w=np.array(table.column("c2w").to_pylist()).reshape(-1, 4, 4))
        save_pairs()
        print(json.dumps({k: out[k] for k in ("recipe", "gauge_lock", "wall_s", "keyframes",
                                             "registered", "map_version")}))
        if cont:
            for w in cont["reloc_windows"]:
                print(f"  t={w['t0']:4d}s: {w['relocs']} relocs, moved median {w['median']:.2%} "
                      f"p90 {w['p90']:.2%} max {w['max']:.2%}, >1% {w['gt1pct']}, >5% {w['gt5pct']}")
            print("  drift since first registered:", json.dumps(cont["drift_since_first_registered"]))
        for r in s.refit_log:
            print(f"  v{r['version']}: {r['seconds']} s, scale {r['scale']:.3f} rot {r['rot_deg']:.2f} deg "
                  f"shift {r['shift']:.1%} | moved med {r['res_median']:.2%} p90 {r['res_p90']:.2%} "
                  f"max {r['res_max']:.2%} rot {r['rot_res_median_deg']:.2f} deg | raw med {r['raw_median']:.2%}")
    finally:
        s.stop()
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
