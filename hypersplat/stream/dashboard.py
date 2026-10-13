"""Live dashboard for a streaming run: 3DGS viewer (viser) on the left, camera feed on the right.

    GET /             split page; the left pane embeds the trainer's viser viewer once training
                      starts (before that it shows the bootstrap progress)
    GET /frame.jpg    newest frame the session received (keyframes flash green); the page polls it
    GET /feed.mjpg    the same as an MJPEG stream (VLC, other clients)
    GET /status.json  session stats + phase + training progress + viewer port
    GET /metrics.json every TensorBoard scalar of the run (<result_dir>/tb) as downsampled series;
                      the page charts them live below the viewer
    GET /depth.jpg    live depth of the camera while the first map is built (with --depth)
    POST /api/stop_recording
                      stop taking camera frames (camera released); training continues on the
                      keyframes collected and then finishes as usual

The camera can only be opened by one process, so the feed is served from the frames the
session already reads - no second capture. Binds to 127.0.0.1 by default: the feed is your
camera, so exposing it on the network is an explicit choice (host="0.0.0.0").
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FEED_FPS = 15
FEED_QUALITY = 80


def _overlay(image: np.ndarray, keyframe: bool) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if keyframe:
        h, w = bgr.shape[:2]
        t = max(3, w // 160)
        cv2.rectangle(bgr, (t // 2, t // 2), (w - t // 2 - 1, h - t // 2 - 1), (80, 200, 60), t)
        cv2.putText(bgr, "KEYFRAME", (2 * t, 9 * t), cv2.FONT_HERSHEY_SIMPLEX, t / 5, (80, 200, 60),
                    max(1, t // 2), cv2.LINE_AA)
    return bgr


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live 3DGS Stream</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --text:#e6e8ec; --muted:#9aa3b2;
          --accent:#5ec27a; --warn:#e0a84a; --bad:#e06464; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f4f5f7; --panel:#ffffff; --line:#dde1e7; --text:#1b1f27; --muted:#5b6474;
            --accent:#2f9a52; --warn:#b7791f; --bad:#c53030; }
  }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; flex-direction:column; background:var(--bg);
         color:var(--text); font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:10px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .badge { padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600;
           border:1px solid currentColor; color:var(--warn); text-transform:uppercase; }
  .badge.tracking, .badge.training, .badge.done { color:var(--accent); }
  .badge.failed { color:var(--bad); }
  #hint { color:var(--muted); }
  main { flex:1; display:flex; min-height:0; }
  #left { flex:3; min-width:0; position:relative; border-right:1px solid var(--line); }
  #left { display:flex; flex-direction:column; }
  #vwrap { position:relative; flex:1; min-height:0; }
  #charts { flex:0 0 36vh; overflow:auto; border-top:1px solid var(--line); background:var(--panel);
            padding:6px 10px 10px; }
  #charts h3 { font-size:11px; margin:6px 0 4px; color:var(--muted); text-transform:uppercase;
               letter-spacing:.06em; }
  .cgrid { display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); gap:6px; }
  .chart { border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
  .chart .t { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); }
  .chart .t b { color:var(--text); font-weight:600; font-variant-numeric:tabular-nums; }
  .chart canvas { width:100%; height:64px; display:block; }
  #left iframe { width:100%; height:100%; border:0; display:none; background:#000; }
  #placeholder { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
                 justify-content:center; gap:10px; color:var(--muted); text-align:center; padding:24px; }
  #depthprev { max-width:92%; max-height:70%; border-radius:6px; }
  .spinner { width:28px; height:28px; border-radius:50%; border:3px solid var(--line);
             border-top-color:var(--accent); animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #right { flex:2; min-width:280px; display:flex; flex-direction:column; gap:12px; padding:12px;
           overflow:auto; background:var(--panel); }
  #feed { width:100%; border-radius:6px; background:#000; aspect-ratio:16/9; object-fit:contain; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:8px; }
  .stat { border:1px solid var(--line); border-radius:6px; padding:8px 10px; }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .stat .v { font-size:18px; font-variant-numeric:tabular-nums; }
  .bar { height:6px; border-radius:3px; background:var(--line); overflow:hidden; }
  .bar > div { height:100%; width:0; background:var(--accent); transition:width .5s; }
  #stoprec { margin-left:auto; padding:6px 14px; border-radius:6px; border:1px solid var(--bad);
             background:transparent; color:var(--bad); font:inherit; font-weight:600; cursor:pointer; }
  #stoprec:hover:not(:disabled) { background:var(--bad); color:var(--panel); }
  #stoprec:disabled { border-color:var(--line); color:var(--muted); cursor:default; }
  .world { color:var(--muted); font-size:12px; }
  .world.done { color:var(--accent); font-weight:600; }
  h2 { font-size:12px; margin:4px 0 0; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
  @media (max-width: 800px) { main { flex-direction:column; } #left { min-height:55vh; border-right:0; } }
</style>
</head>
<body>
<header>
  <h1>Live 3DGS stream</h1>
  <span id="phase" class="badge">starting</span>
  <span id="hint">Connecting...</span>
  <button id="stoprec" type="button" hidden
          title="Stop taking camera frames; training continues on the keyframes collected">&#9632; Stop recording</button>
</header>
<main>
  <section id="left">
    <div id="vwrap">
    <iframe id="viewer" title="3DGS viewer"></iframe>
    <div id="placeholder"><div class="spinner" id="spinner"></div>
      <img id="depthprev" alt="Live depth preview" hidden>
      <div id="ph-text">The 3D viewer opens when training starts.</div></div>
    </div>
    <div id="charts"><div class="world" id="charts-empty">Loss charts (TensorBoard) appear once training logs.</div></div>
  </section>
  <section id="right">
    <h2>Camera</h2>
    <img id="feed" alt="Live camera feed">
    <h2>Training</h2>
    <div class="bar"><div id="progress"></div></div>
    <div id="world" class="world"></div>
    <div class="grid" id="train"></div>
    <h2>Stream</h2>
    <div class="grid" id="stream"></div>
  </section>
</main>
<script>
const fmt = v => v === null || v === undefined ? "-" : (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(v < 10 ? 3 : 1) : v);
function fill(id, pairs) {
  document.getElementById(id).innerHTML = pairs.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${fmt(v)}</div></div>`).join("");
}
// Camera: poll the newest frame and swap it in once decoded (no flicker, works in every
// browser, recovers by itself if the run restarts)
const feed = document.getElementById("feed");
let feedUrl = null;
async function pollFrame() {
  let delay = 66;
  try {
    const r = await fetch("/frame.jpg", {cache: "no-store"});
    if (r.status === 200) {
      const url = URL.createObjectURL(await r.blob());
      feed.src = url;
      if (feedUrl) URL.revokeObjectURL(feedUrl);
      feedUrl = url;
    } else { delay = 500; }
  } catch (e) { delay = 2000; }
  setTimeout(pollFrame, delay);
}
window.addEventListener("load", pollFrame);
const stopBtn = document.getElementById("stoprec");
stopBtn.addEventListener("click", async () => {
  if (!confirm("Stop recording? The camera is released and training finishes on the keyframes collected so far.")) return;
  stopBtn.disabled = true; stopBtn.textContent = "Stopping...";
  try { await fetch("/api/stop_recording", {method: "POST"}); } catch (e) {}
});
// Live depth preview in the left pane until the 3D viewer takes over
const depthImg = document.getElementById("depthprev");
let depthUrl = null;
async function pollDepth() {
  if (viewerSet) return;
  try {
    const r = await fetch("/depth.jpg", {cache: "no-store"});
    if (r.status === 200) {
      const url = URL.createObjectURL(await r.blob());
      depthImg.src = url; depthImg.hidden = false;
      document.getElementById("spinner").hidden = true;
      if (depthUrl) URL.revokeObjectURL(depthUrl);
      depthUrl = url;
    }
  } catch (e) {}
  setTimeout(pollDepth, 350);
}
pollDepth();
// Live loss charts from the run's TensorBoard scalars, grouped by tag prefix
const charts = {};
function drawChart(cv, pts, color) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  if (pts.length < 2) return;
  let x0 = pts[0][0], x1 = pts[pts.length - 1][0], lo = Infinity, hi = -Infinity;
  for (const p of pts) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = 3, X = x => pad + (w - 2 * pad) * (x - x0) / Math.max(1, x1 - x0),
        Y = y => h - pad - (h - 2 * pad) * (y - lo) / (hi - lo);
  g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
  pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])));
  g.stroke();
}
async function pollMetrics() {
  try {
    const m = await (await fetch("/metrics.json", {cache: "no-store"})).json();
    const tags = Object.keys(m.scalars || {}).sort();
    const box = document.getElementById("charts");
    if (tags.length) document.getElementById("charts-empty")?.remove();
    const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#5ec27a";
    for (const tag of tags) {
      let c = charts[tag];
      if (!c) {
        const grp = tag.includes("/") ? tag.split("/")[0] : "other";
        let sec = document.getElementById("grp-" + grp);
        if (!sec) {
          box.insertAdjacentHTML("beforeend", `<h3>${grp}</h3><div class="cgrid" id="grp-${grp}"></div>`);
          sec = document.getElementById("grp-" + grp);
        }
        const el = document.createElement("div"); el.className = "chart";
        el.innerHTML = `<div class="t"><span></span><b></b></div><canvas></canvas>`;
        el.firstChild.firstChild.textContent = tag.slice(tag.indexOf("/") + 1);
        sec.appendChild(el);
        c = charts[tag] = {cv: el.querySelector("canvas"), val: el.querySelector("b")};
      }
      const pts = m.scalars[tag];
      c.val.textContent = pts.length ? fmt(pts[pts.length - 1][1]) : "-";
      drawChart(c.cv, pts, color);
    }
  } catch (e) {}
  setTimeout(pollMetrics, 3000);
}
pollMetrics();
let viewerSet = false;
async function tick() {
  try {
    const s = await (await fetch("/status.json", {cache: "no-store"})).json();
    const t = s.training || {};
    const phase = s.phase === "failed" ? "failed" : t.done ? "done" : t.step !== undefined ? "training" : s.phase;
    const badge = document.getElementById("phase");
    badge.textContent = phase; badge.className = "badge " + phase;
    document.getElementById("hint").textContent = t.done ? "Training finished; outputs saved" : s.hint;
    document.getElementById("ph-text").textContent = s.hint;
    stopBtn.hidden = !s.live;
    if (s.source_ended || s.recording_stopped) {
      stopBtn.disabled = true; stopBtn.textContent = "Recording stopped";
    }
    if (s.viewer_port && !viewerSet) {
      const f = document.getElementById("viewer");
      f.src = `${location.protocol}//${location.hostname}:${s.viewer_port}`;
      f.style.display = "block"; document.getElementById("placeholder").style.display = "none";
      viewerSet = true;
    }
    const w = document.getElementById("world");
    w.textContent = t.world ? "World: " + t.world : "";
    w.className = "world" + (t.world && t.world.startsWith("converged") ? " done" : "");
    if (t.max_steps) document.getElementById("progress").style.width = (100 * (t.step || 0) / t.max_steps) + "%";
    fill("train", [["Step", t.step !== undefined ? `${t.step}/${t.max_steps}` : null], ["Gaussians", t.num_gs],
                   ["Loss", t.loss], ["Difix rounds", t.difix_rounds], ["Held-out PSNR", t.psnr]]);
    fill("stream", [["Frames", s.frames_read], ["Dropped", s.frames_dropped], ["Keyframes", s.keyframes],
                    ["Registered", s.registered], ["Pseudo views", s.pseudo], ["Map version", s.map_version],
                    ["Store MB", `${fmt(s.store_mb)} / ${s.budget_mb}`], ["Localize ms", s.localize_ms]]
                   .concat(s.depth_maps === null || s.depth_maps === undefined ? [] :
                     [["Depth maps", `${s.depth_maps} (${s.depth_rejected} rejected)`],
                      ["Depth error", s.depth_rel_err === null ? null : (100 * s.depth_rel_err).toFixed(1) + " %"],
                      ["Depth ms", s.depth_ms]]));
  } catch (e) {
    document.getElementById("hint").textContent = "Run ended (dashboard offline)";
  }
  setTimeout(tick, 1000);
}
tick();
</script>
</body>
</html>
"""


class Dashboard:
    def __init__(self, session, port: int = 8090, host: str = "127.0.0.1", tb_dir: Optional[str] = None):
        self.session = session
        self.tb_dir = tb_dir
        self._tb = None
        self._tb_cache = (0.0, {})
        self._tb_lock = threading.Lock()
        self.host, self.port = host, port
        self.viewer_port: Optional[int] = None
        self.training: Dict = {}
        self._stopped = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._jpeg_cache = None
        self.depth_preview: Optional[bytes] = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{host}:{self.port}"

    def update_training(self, **info):
        self.training.update({k: v for k, v in info.items() if v is not None})

    def status(self) -> Dict:
        stats = self.session.stats()
        phase, hint = self.session.phase()
        return {**stats, "phase": phase, "hint": hint, "viewer_port": self.viewer_port,
                "training": dict(self.training)}

    def metrics(self, max_points: int = 300) -> Dict:
        """Every TensorBoard scalar of the run as [[step, value], ...] (evenly thinned to
        max_points); re-read at most every 2 s."""
        with self._tb_lock:
            t, cached = self._tb_cache
            if time.time() - t < 2.0 or not self.tb_dir:
                return cached
            try:
                if self._tb is None:
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    self._tb = EventAccumulator(self.tb_dir, size_guidance={"scalars": 0})
                self._tb.Reload()
                out = {}
                for tag in self._tb.Tags().get("scalars", []):
                    ev = self._tb.Scalars(tag)
                    stride = max(1, len(ev) // max_points)
                    pts = ev[::stride]
                    if pts[-1] is not ev[-1]:
                        pts = pts + [ev[-1]]
                    out[tag] = [[e.step, e.value] for e in pts]
                cached = {"scalars": out}
            except Exception as e:  # tensorboard missing / no event file yet
                cached = {"scalars": cached.get("scalars", {}), "error": str(e)}
            self._tb_cache = (time.time(), cached)
            return cached

    def frame_jpeg(self) -> Optional[bytes]:
        """Newest source frame as JPEG (cached per frame), or None before the first frame."""
        item = self.session.latest_frame()
        if item is None:
            return None
        key = (item[3], self.session.recently_accepted())
        if self._jpeg_cache is None or self._jpeg_cache[0] != key:
            ok, jpg = cv2.imencode(".jpg", _overlay(item[0], key[1]), [cv2.IMWRITE_JPEG_QUALITY, FEED_QUALITY])
            if not ok:
                return None
            self._jpeg_cache = (key, jpg.tobytes())
        return self._jpeg_cache[1]

    def start(self) -> "Dashboard":
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the training log clean
                pass

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path.split("?")[0] == "/api/stop_recording":
                    stopped = dash.session.stop_recording("dashboard")
                    self._send(200, "application/json", json.dumps({"stopped": stopped}).encode())
                else:
                    self._send(404, "text/plain", b"not found")

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif path == "/status.json":
                    self._send(200, "application/json", json.dumps(dash.status(), default=str).encode())
                elif path == "/metrics.json":
                    self._send(200, "application/json", json.dumps(dash.metrics(), default=str).encode())
                elif path == "/frame.jpg":
                    jpg = dash.frame_jpeg()
                    if jpg is None:
                        self._send(204, "image/jpeg", b"")
                    else:
                        self._send(200, "image/jpeg", jpg)
                elif path == "/depth.jpg":
                    jpg = dash.depth_preview
                    self._send(200, "image/jpeg", jpg) if jpg else self._send(204, "image/jpeg", b"")
                elif path == "/feed.mjpg":
                    self._feed()
                else:
                    self._send(404, "text/plain", b"not found")

            def _feed(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                # Every part is closed by the next boundary right away: browsers only draw a
                # part once they see that boundary, so a stalled source would never show its
                # last frame. The last frame is also re-sent every second as a keepalive.
                last, sent_at = -1, 0.0
                try:
                    self.wfile.write(b"--frame\r\n")
                    while not dash._stopped.is_set():
                        item = dash.session.latest_frame()
                        now = time.time()
                        if item is not None and (item[3] != last or now - sent_at > 1.0):
                            last, sent_at = item[3], now
                            ok, jpg = cv2.imencode(".jpg", _overlay(item[0], dash.session.recently_accepted()),
                                                   [cv2.IMWRITE_JPEG_QUALITY, FEED_QUALITY])
                            if ok:
                                data = jpg.tobytes()
                                self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(data)).encode() + b"\r\n\r\n" + data
                                                 + b"\r\n--frame\r\n")
                                self.wfile.flush()
                        time.sleep(1.0 / FEED_FPS)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="stream-dashboard", daemon=True).start()
        logger.info(f"[Stream] Dashboard: {self.url}")
        return self

    def stop(self):
        self._stopped.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
