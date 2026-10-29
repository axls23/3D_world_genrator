"""Live dashboard for a streaming run: 3DGS viewer (viser) on the left, camera feed on the right.

    GET /             split page; the left pane embeds the trainer's viser viewer once training
                      starts (before that it shows the bootstrap progress)
    GET /frame.jpg    newest frame the session received (keyframes flash green); the page polls it
    GET /feed.mjpg    the same as an MJPEG stream (VLC, other clients)
    GET /status.json  session stats + phase + training progress + viewer port
    GET /poses.json   registered camera centers for the trajectory graphic
    GET /metrics.json every TensorBoard scalar of the run (<result_dir>/tb) as downsampled series;
                      the page charts them live below the viewer
    GET /depth.jpg    newest camera depth heatmap (with --depth)
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
  :root { color-scheme:dark; --bg:#0a0f17; --panel:#111a27; --panel2:#152131; --line:#253448;
          --text:#e8eef5; --muted:#8999ad; --accent:#49d6b3; --blue:#68a9ff;
          --warn:#f3b95f; --bad:#ff6b78; --shadow:0 12px 32px #0003; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; background:radial-gradient(ellipse at 12% -15%,#17334a 0,transparent 35%),var(--bg);
         color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; align-items:center; gap:14px; min-height:72px; padding:12px 24px;
           border-bottom:1px solid var(--line); background:#0b121de8; position:sticky; top:0; z-index:5;
           backdrop-filter:blur(16px); }
  .brand { display:flex; align-items:center; gap:11px; min-width:245px; }
  .brand-mark { width:34px; height:34px; display:grid; place-items:center; border-radius:11px;
                color:var(--accent); background:#49d6b31a; border:1px solid #49d6b344; font-size:20px; }
  header h1 { font-size:15px; margin:0; font-weight:700; letter-spacing:.01em; }
  .eyebrow,.section-label { color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase;
                            letter-spacing:.13em; }
  .run-meta { color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .badge { display:inline-flex; align-items:center; gap:7px; padding:5px 10px; border-radius:999px;
           font-size:10px; font-weight:800; letter-spacing:.08em; border:1px solid #f3b95f55;
           color:var(--warn); background:#f3b95f12; text-transform:uppercase; }
  .badge::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }
  .badge.tracking,.badge.training,.badge.done { color:var(--accent); border-color:#49d6b344; background:#49d6b30f; }
  .badge.failed,.badge.reconnecting { color:var(--bad); border-color:#ff6b7855; background:#ff6b7812; }
  #hint { color:#b6c4d5; font-size:12px; flex:1; min-width:130px; }
  #stoprec { padding:8px 13px; border-radius:9px; border:1px solid #ff6b7866;
             background:#ff6b7812; color:#ff8891; font:inherit; font-size:12px; font-weight:700; cursor:pointer; }
  #stoprec:hover:not(:disabled) { background:var(--bad); color:#10151d; }
  #stoprec:disabled { border-color:var(--line); color:var(--muted); background:transparent; cursor:default; }
  #overview { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; padding:18px 24px 12px; }
  .kpi,.panel { background:linear-gradient(145deg,#142031,#101925 72%); border:1px solid var(--line);
                border-radius:14px; box-shadow:var(--shadow); }
  .kpi { padding:14px 16px; position:relative; overflow:hidden; }
  .kpi::after { content:""; position:absolute; right:-24px; top:-32px; width:90px; height:90px;
                border-radius:50%; background:var(--glow,#49d6b3); opacity:.08; filter:blur(2px); }
  .kpi-head { display:flex; align-items:center; justify-content:space-between; color:var(--muted);
              font-size:10px; font-weight:750; text-transform:uppercase; letter-spacing:.1em; }
  .kpi-icon { color:var(--tone,var(--accent)); font-size:15px; }
  .kpi-value { margin-top:7px; font-size:25px; line-height:1.1; font-weight:720; letter-spacing:-.04em;
               font-variant-numeric:tabular-nums; }
  .kpi-note { margin-top:6px; color:var(--muted); font-size:11px; }
  main { display:grid; grid-template-columns:minmax(0,1.7fr) minmax(320px,.8fr); gap:14px;
         padding:4px 24px 24px; align-items:start; }
  #left,#right { min-width:0; display:flex; flex-direction:column; gap:14px; }
  .panel { padding:15px; min-width:0; }
  .panel-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
  .panel-title { font-size:13px; font-weight:700; margin:0; }
  .panel-subtitle { color:var(--muted); font-size:11px; margin-top:2px; }
  .live-tag { display:inline-flex; align-items:center; gap:6px; color:#89a0b8; font-size:10px;
              font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
  .live-tag::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--accent);
                      box-shadow:0 0 10px #49d6b3aa; }
  #vwrap { position:relative; min-height:350px; height:min(52vh,560px); overflow:hidden; border-radius:10px;
           border:1px solid #26384b; background:radial-gradient(ellipse at center,#18283a,#080d14 75%); }
  #left iframe { width:100%; height:100%; border:0; display:none; background:#080d14; }
  .viewer-link { color:var(--blue); font-size:10px; font-weight:700; text-decoration:none; white-space:nowrap; }
  .viewer-link:hover { text-decoration:underline; }
  #placeholder { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
                 justify-content:center; gap:10px; color:var(--muted); text-align:center; padding:24px; }

  .spinner { width:28px; height:28px; border-radius:50%; border:3px solid var(--line);
             border-top-color:var(--accent); animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .two-up { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(220px,.85fr); gap:12px; }
  #posemap { width:100%; height:190px; display:block; border:1px solid #243449; border-radius:10px;
             background:linear-gradient(180deg,#0c1521,#101b29); }
  .map-legend { display:flex; gap:12px; color:var(--muted); font-size:10px; margin-top:8px; }
  .legend-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; background:var(--accent); }
  .legend-dot.bad { background:var(--bad); }
  .camera-frame { position:relative; background:#080d14; border-radius:10px; overflow:hidden;
                  border:1px solid #26384b; }
  #feed { width:100%; display:block; border-radius:10px; aspect-ratio:16/9; object-fit:contain; }
  #depth-overlay { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
                   opacity:.48; pointer-events:none; }
  .depth-toggle { padding:6px 9px; border:1px solid var(--line); border-radius:8px;
                  background:#152131; color:var(--muted); font:inherit; font-size:10px;
                  font-weight:700; cursor:pointer; }
  .depth-toggle[aria-pressed="true"] { border-color:#f3b95f88; color:#ffd27a; }
  .depth-toggle:disabled { opacity:.55; cursor:default; }
  #feed-placeholder { position:absolute; inset:0; display:grid; place-items:center; color:var(--muted);
                      font-size:11px; background:radial-gradient(ellipse at center,#172538,#080d14 75%); }
  .camera-caption { display:flex; justify-content:space-between; gap:8px; color:var(--muted);
                    font-size:10px; padding-top:8px; }
  .camera-caption strong { color:var(--text); font-weight:600; }
  .progress-track { height:7px; border-radius:9px; background:#202e40; overflow:hidden; margin:10px 0 6px; }
  .progress-track .bar { height:100%; }
  .bar > div { height:100%; width:0; background:linear-gradient(90deg,#35bfa2,#68a9ff);
               border-radius:9px; transition:width .45s; box-shadow:0 0 12px #49d6b355; }
  .progress-meta { display:flex; justify-content:space-between; color:var(--muted); font-size:10px; }
  .metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
  .stat { border:1px solid #26374b; border-radius:10px; padding:9px 10px; background:#0d1622a8; min-width:0; }
  .stat .k { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.09em; }
  .stat .v { margin-top:4px; font-size:17px; font-weight:650; font-variant-numeric:tabular-nums;
             overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #world { color:#b8c7d7; font-size:11px; }
  .world.done { color:var(--accent); font-weight:700; }
  .flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); align-items:stretch; gap:7px; }
  .flow-step { text-align:center; border:1px solid #26374b; background:#0d1622a8; border-radius:10px;
               padding:10px 5px; min-width:0; }
  .flow-step strong { display:block; font-size:10px; font-weight:700; }
  .flow-step span { display:block; margin-top:4px; color:var(--muted); font-size:9px; }
  .flow-arrow { display:none; }
  .quality { display:flex; align-items:flex-start; gap:10px; padding:11px 12px; border-radius:10px;
             border:1px solid #f3b95f44; background:#f3b95f0b; }
  .quality-icon { font-size:17px; color:var(--warn); }
  .quality strong { display:block; font-size:11px; }
  .quality p { margin:3px 0 0; color:var(--muted); font-size:10px; line-height:1.5; }
  #charts { overflow:auto; background:transparent; }
  #charts h3 { font-size:10px; margin:12px 0 7px; color:var(--muted); text-transform:uppercase;
               letter-spacing:.1em; }
  #charts h3:first-child { margin-top:0; }
  .cgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:8px; }
  .chart { border:1px solid #26374b; border-radius:10px; padding:8px 10px; background:#0d1622a8; }
  .chart .t { display:flex; justify-content:space-between; font-size:10px; color:var(--muted); }
  .chart .t b { color:var(--text); font-weight:650; font-variant-numeric:tabular-nums; }
  .chart canvas { width:100%; height:76px; display:block; }
  .empty-note { color:var(--muted); font-size:11px; padding:13px; border:1px dashed #34465b; border-radius:9px; }
  @media (max-width: 980px) {
    main { grid-template-columns:minmax(0,1fr) 320px; padding-left:16px; padding-right:16px; }
    #overview { padding-left:16px; padding-right:16px; }
  }
  @media (max-width: 760px) {
    header { padding:10px 14px; flex-wrap:wrap; }
    .brand { min-width:0; }
    #hint { order:4; flex-basis:100%; }
    #overview { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; padding:12px 12px 8px; }
    main { grid-template-columns:minmax(0,1fr); gap:10px; padding:4px 12px 18px; }
    #vwrap { height:48vh; min-height:280px; }
    .two-up { grid-template-columns:minmax(0,1fr); }
  }
  @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms!important; transition-duration:.01ms!important; } }
</style>
</head>
<body>
<header>
  <div class="brand"><div class="brand-mark" aria-hidden="true">&#9672;</div><div>
    <div class="eyebrow">HyperSplat / telemetry</div><h1>Live reconstruction</h1></div></div>
  <span id="phase" class="badge">starting</span>
  <span id="source-name" class="run-meta">Connecting to source</span>
  <span id="hint">Connecting...</span>
  <button id="stoprec" type="button" hidden
          title="Stop taking camera frames; training continues on the keyframes collected">&#9632; Stop capture</button>
</header>
<section id="overview" aria-label="Run overview">
  <article class="kpi" style="--tone:var(--blue);--glow:var(--blue)"><div class="kpi-head">Frames received <span class="kpi-icon">&#9638;</span></div>
    <div class="kpi-value" id="kpi-frames">--</div><div class="kpi-note" id="kpi-dropped">Waiting for source</div></article>
  <article class="kpi"><div class="kpi-head">Keyframe yield <span class="kpi-icon">&#9673;</span></div>
    <div class="kpi-value" id="kpi-yield">--</div><div class="kpi-note" id="kpi-keyframes">Accepted / offered</div></article>
  <article class="kpi" style="--tone:var(--blue);--glow:var(--blue)"><div class="kpi-head">Map coverage <span class="kpi-icon">&#9678;</span></div>
    <div class="kpi-value" id="kpi-coverage">--</div><div class="kpi-note" id="kpi-map">Registered views</div></article>
  <article class="kpi" style="--tone:var(--warn);--glow:var(--warn)"><div class="kpi-head">Training loss <span class="kpi-icon">&#8767;</span></div>
    <div class="kpi-value" id="kpi-loss">--</div><div class="kpi-note" id="kpi-step">Waiting for training</div></article>
</section>
<main>
  <section id="left" aria-label="Reconstruction and telemetry">
    <article class="panel">
      <div class="panel-head"><div><div class="section-label">3D scene</div><h2 class="panel-title">Reconstruction viewport</h2>
        <div class="panel-subtitle">Live Gaussian scene and camera trajectory</div></div>
        <div style="display:flex;align-items:center;gap:12px"><span class="live-tag">Live view</span>
          <a id="viewer-link" class="viewer-link" target="_blank" rel="noopener" hidden>Open viewer &#8599;</a></div></div>
      <div id="vwrap"><iframe id="viewer" title="3DGS viewer"></iframe>
        <div id="placeholder"><div class="spinner" id="spinner"></div>
          <div id="ph-text">The 3D viewer opens when mapping begins.</div></div></div>
      <div class="two-up" style="margin-top:12px">
        <div><div class="panel-head"><div><div class="section-label">Pose telemetry</div><div class="panel-subtitle">Registered camera centers · X/Z projection</div></div></div>
          <canvas id="posemap" aria-label="Registered camera trajectory"></canvas>
          <div class="map-legend"><span><i class="legend-dot"></i>Registered pose</span><span><i class="legend-dot bad"></i>Flagged pose</span></div></div>
        <div><div class="panel-head"><div><div class="section-label">Signal integrity</div><div class="panel-subtitle">Data quality and model status</div></div></div>
          <div class="quality" id="anomaly-status"><span class="quality-icon">&#9888;</span><div>
            <strong>Anomaly model not connected</strong><p>This run does not emit anomaly scores yet. Red markers will appear here when an offline detector is integrated.</p>
          </div></div><div class="metric-grid" id="quality-stats" style="margin-top:8px"></div></div>
      </div>
    </article>
    <article class="panel">
      <div class="panel-head"><div><div class="section-label">Training telemetry</div><h2 class="panel-title">Optimization trends</h2></div>
        <span class="panel-subtitle">Refreshes from TensorBoard</span></div>
      <div id="charts"><div class="empty-note" id="charts-empty">Training curves appear when the first scalar events are written.</div></div>
    </article>
  </section>
  <aside id="right" aria-label="Live run status">
    <article class="panel"><div class="panel-head"><div><div class="section-label">Input</div><h2 class="panel-title">Camera feed</h2></div>
      <div style="display:flex;align-items:center;gap:10px"><button id="depth-toggle" class="depth-toggle"
        type="button" aria-pressed="true" disabled>Depth heatmap</button><span class="live-tag">Source</span></div></div>
      <div class="camera-frame"><img id="feed" alt="Live camera feed">
        <img id="depth-overlay" alt="Depth heatmap overlay" hidden>
        <div id="feed-placeholder" aria-live="polite">Waiting for source frames</div></div>
      <div class="camera-caption"><span>Latest frame</span><strong id="frame-status">Waiting for frames</strong></div></article>
    <article class="panel"><div class="panel-head"><div><div class="section-label">Optimization</div><h2 class="panel-title">Training progress</h2></div>
      <span class="panel-subtitle" id="step-label">-- / --</span></div>
      <div class="progress-track"><div class="bar"><div id="progress"></div></div></div>
      <div class="progress-meta"><span id="world">Initializing model</span><span id="progress-pct">--</span></div>
      <div class="metric-grid" id="train" style="margin-top:12px"></div></article>
    <article class="panel"><div class="panel-head"><div><div class="section-label">Event pipeline</div><h2 class="panel-title">Data flow health</h2></div></div>
      <div class="flow">
        <div class="flow-step"><strong>CAPTURE</strong><span id="flow-capture">0 frames</span></div>
        <div class="flow-step"><strong>KEYFRAMES</strong><span id="flow-keyframes">0 accepted</span></div>
        <div class="flow-step"><strong>ACE MAP</strong><span id="flow-map">Not ready</span></div>
        <div class="flow-step"><strong>3D TRAIN</strong><span id="flow-train">Waiting</span></div>
      </div><div class="metric-grid" id="stream" style="margin-top:9px"></div></article>
  </aside>
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
const feedPlaceholder = document.getElementById("feed-placeholder");
feed.addEventListener("load", () => { feedPlaceholder.hidden = true; });
feed.addEventListener("error", () => { feedPlaceholder.hidden = false; });
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
const depthToggle = document.getElementById("depth-toggle");
const depthOverlay = document.getElementById("depth-overlay");
depthToggle.addEventListener("click", () => {
  const enabled = depthToggle.getAttribute("aria-pressed") !== "true";
  depthToggle.setAttribute("aria-pressed", String(enabled));
  depthToggle.textContent = enabled ? "Depth heatmap" : "Show depth";
  depthOverlay.hidden = !enabled || !depthOverlay.src;
});
depthOverlay.addEventListener("load", () => {
  depthToggle.disabled = false;
  depthOverlay.hidden = depthToggle.getAttribute("aria-pressed") !== "true";
});
depthOverlay.addEventListener("error", () => { depthOverlay.hidden = true; });
const stopBtn = document.getElementById("stoprec");
stopBtn.addEventListener("click", async () => {
  if (!confirm("Stop recording? The camera is released and training finishes on the keyframes collected so far.")) return;
  stopBtn.disabled = true; stopBtn.textContent = "Stopping...";
  try { await fetch("/api/stop_recording", {method: "POST"}); } catch (e) {}
});
// Keep the depth heatmap available as a toggleable camera overlay during training.
let viewerSet = false;
let depthUrl = null;
async function pollDepth() {
  let delay = 1000;
  try {
    const r = await fetch("/depth.jpg", {cache: "no-store"});
    if (r.status === 200) {
      const url = URL.createObjectURL(await r.blob());
      depthOverlay.src = url;
      if (depthUrl) URL.revokeObjectURL(depthUrl);
      depthUrl = url;
    } else { delay = 2000; }
  } catch (e) {}
  setTimeout(pollDepth, delay);
}
pollDepth();
// Live loss charts from the run's TensorBoard scalars, grouped by tag prefix
const charts = {};
const latestScalars = {};
const scalarSeries = {};
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
      if (pts.length) {
        latestScalars[tag] = pts[pts.length - 1][1];
        scalarSeries[tag] = pts;
      }
      drawChart(c.cv, pts, color);
    }
  } catch (e) {}
  setTimeout(pollMetrics, 3000);
}
pollMetrics();

function drawPoseMap(poses) {
  const cv = document.getElementById("posemap"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = "#26374b"; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const x = w * i / 4, y = h * i / 4;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
  g.fillStyle = "#71839a"; g.font = "9px system-ui";
  g.fillText("X", w - 13, h - 7); g.fillText("Z", 7, 12);
  if (!poses.length) {
    g.fillStyle = "#8999ad"; g.textAlign = "center";
    g.fillText("Waiting for registered camera poses", w / 2, h / 2);
    return;
  }
  const xs = poses.map(p => p.x), zs = poses.map(p => p.z);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), z0 = Math.min(...zs), z1 = Math.max(...zs);
  if (x0 === x1) { x0 -= .5; x1 += .5; }
  if (z0 === z1) { z0 -= .5; z1 += .5; }
  const pad = 18, px = p => pad + (w - 2 * pad) * (p.x - x0) / (x1 - x0),
        py = p => h - pad - (h - 2 * pad) * (p.z - z0) / (z1 - z0);
  g.strokeStyle = "#49d6b3"; g.lineWidth = 2; g.beginPath();
  poses.forEach((p, i) => i ? g.lineTo(px(p), py(p)) : g.moveTo(px(p), py(p)));
  g.stroke();
  poses.forEach((p, i) => {
    g.beginPath(); g.arc(px(p), py(p), i === poses.length - 1 ? 4.5 : 3, 0, Math.PI * 2);
    g.fillStyle = p.flagged ? "#ff6b78" : i === poses.length - 1 ? "#68a9ff" : "#49d6b3";
    g.fill();
  });
}
async function pollPoses() {
  try {
    const r = await fetch("/poses.json", {cache: "no-store"});
    if (r.ok) drawPoseMap((await r.json()).poses || []);
  } catch (e) {}
  setTimeout(pollPoses, 2000);
}
pollPoses();

async function tick() {
  try {
    const s = await (await fetch("/status.json", {cache: "no-store"})).json();
    const t = s.training || {};
    const phase = s.phase === "failed" ? "failed" : t.done ? "done" : t.step !== undefined ? "training" : s.phase;
    const badge = document.getElementById("phase");
    badge.textContent = phase; badge.className = "badge " + phase;
    document.getElementById("hint").textContent = t.done ? "Training finished; outputs saved" : s.hint;
    document.getElementById("ph-text").textContent = s.hint;
    const source = String(s.source || "stream source").split(/[\\/]/).pop();
    document.getElementById("source-name").textContent = `${s.kind || "source"} · ${source}`;
    stopBtn.hidden = !s.live;
    if (s.source_ended || s.recording_stopped) {
      stopBtn.disabled = true; stopBtn.textContent = "Capture stopped";
    }
    if (s.viewer_port && !viewerSet) {
      const f = document.getElementById("viewer");
      f.src = `${location.protocol}//${location.hostname}:${s.viewer_port}`;
      f.style.display = "block"; document.getElementById("placeholder").style.display = "none";
      const link = document.getElementById("viewer-link");
      link.href = f.src; link.hidden = false;
      viewerSet = true;
    }
    const w = document.getElementById("world");
    w.textContent = t.world ? "World: " + t.world : "";
    w.className = "world" + (t.world && t.world.startsWith("converged") ? " done" : "");
    const pct = t.max_steps ? Math.min(100, 100 * (t.step || 0) / t.max_steps) : 0;
    document.getElementById("progress").style.width = pct + "%";
    document.getElementById("progress-pct").textContent = t.max_steps ? `${pct.toFixed(0)}%` : "--";
    document.getElementById("step-label").textContent = t.max_steps ? `${fmt(t.step || 0)} / ${fmt(t.max_steps)}` : "Awaiting map";
    document.getElementById("frame-status").textContent = s.frames_read ? `${fmt(s.frames_read)} frames · ${fmt(s.frames_dropped || 0)} dropped` : "Waiting for frames";
    document.getElementById("kpi-frames").textContent = fmt(s.frames_read || 0);
    document.getElementById("kpi-dropped").textContent = `${fmt(s.frames_dropped || 0)} dropped · ${fmt(s.reconnects || 0)} reconnects`;
    const yieldPct = s.keyframes_offered ? 100 * s.keyframes_accepted / s.keyframes_offered : null;
    document.getElementById("kpi-yield").textContent = yieldPct === null ? "--" : `${yieldPct.toFixed(1)}%`;
    document.getElementById("kpi-keyframes").textContent = `${fmt(s.keyframes_accepted || 0)} accepted / ${fmt(s.keyframes_offered || 0)} offered`;
    const coverage = s.keyframes ? 100 * (s.registered || 0) / s.keyframes : null;
    document.getElementById("kpi-coverage").textContent = coverage === null ? "--" : `${coverage.toFixed(0)}%`;
    document.getElementById("kpi-map").textContent = `${fmt(s.registered || 0)} registered · map v${fmt(s.map_version || 0)}`;
    document.getElementById("kpi-loss").textContent = t.loss === undefined ? "--" : fmt(t.loss);
    document.getElementById("kpi-step").textContent = t.step === undefined ? "Waiting for training" : `Step ${fmt(t.step)} of ${fmt(t.max_steps)}`;
    document.getElementById("flow-capture").textContent = `${fmt(s.frames_read || 0)} frames`;
    document.getElementById("flow-keyframes").textContent = `${fmt(s.keyframes_accepted || 0)} accepted`;
    document.getElementById("flow-map").textContent = s.map_version ? `Version ${s.map_version}` : "Bootstrapping";
    document.getElementById("flow-train").textContent = t.step === undefined ? "Waiting" : `Step ${fmt(t.step)}`;
    fill("train", [["Step", t.step !== undefined ? `${t.step}/${t.max_steps}` : null], ["Gaussians", t.num_gs],
                   ["Loss", t.loss], ["Difix rounds", t.difix_rounds], ["Held-out PSNR", t.psnr]]);
    fill("stream", [["Frames", s.frames_read], ["Dropped", s.frames_dropped], ["Keyframes", s.keyframes],
                    ["Registered", s.registered], ["Pseudo views", s.pseudo], ["Map version", s.map_version],
                    ["Store MB", `${fmt(s.store_mb)} / ${s.budget_mb}`], ["Localize ms", s.localize_ms]]);
    const lossSeries = scalarSeries["train/loss"] || [];
    const recentLoss = lossSeries.slice(-30).map(point => point[1]);
    const priorLoss = lossSeries.slice(-60, -30).map(point => point[1]);
    const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    const rollingLoss = mean(recentLoss), previousLoss = mean(priorLoss);
    const lossTrend = rollingLoss === null || previousLoss === null ? null
      : `${rollingLoss <= previousLoss ? "Improving" : "Rising"} (${fmt(rollingLoss - previousLoss)})`;
    const gsSeries = scalarSeries["train/num_GS"] || [];
    const gsWindow = gsSeries.slice(-30);
    const gsGrowth = gsWindow.length > 1 && gsWindow[gsWindow.length - 1][0] > gsWindow[0][0]
      ? 100 * (gsWindow[gsWindow.length - 1][1] - gsWindow[0][1]) /
        (gsWindow[gsWindow.length - 1][0] - gsWindow[0][0]) : null;
    const gpuSeries = (scalarSeries["train/mem"] || []).map(point => point[1]);
    const peakGpu = gpuSeries.length ? Math.max(...gpuSeries) : null;
    const quality = [["Unregistered", s.unregistered], ["Pending", s.pending],
                     ["Depth rejected", s.depth_rejected], ["Depth backlog", s.depth_backlog],
                     ["Rolling loss", rollingLoss], ["Best loss", lossSeries.length
                       ? Math.min(...lossSeries.map(point => point[1])) : null],
                     ["Loss trend", lossTrend], ["Gaussian growth / 100 steps", gsGrowth],
                     ["Peak GPU memory", peakGpu === null ? null : `${fmt(peakGpu)} GB`]];
    fill("quality-stats", quality);
    const anomalies = s.anomaly_flags || [];
    if (anomalies.length) {
      const box = document.getElementById("anomaly-status");
      box.style.borderColor = "#ff6b7866"; box.style.background = "#ff6b7812";
      box.querySelector("strong").textContent = `${anomalies.length} flagged event${anomalies.length === 1 ? "" : "s"}`;
      box.querySelector("p").textContent = anomalies[0].reason || "Review flagged pose or training events.";
    }
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

    def pose_path(self) -> Dict:
        store = getattr(self.session, "store", None)
        if store is None:
            return {"map_version": 0, "poses": []}
        poses = []
        for keyframe in store.registered():
            if keyframe.w2c is None:
                continue
            c2w = np.linalg.inv(keyframe.w2c)
            to_trainer = getattr(self.session, "to_trainer", None)
            if callable(to_trainer):
                c2w = to_trainer(c2w)
            x, y, z = c2w[:3, 3]
            poses.append({"kid": keyframe.kid, "name": keyframe.name, "x": float(x), "y": float(y),
                          "z": float(z), "confidence": float(keyframe.conf), "flagged": False})
        return {"map_version": int(getattr(self.session, "map_version", 0)), "poses": poses}

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
                elif path == "/poses.json":
                    self._send(200, "application/json", json.dumps(dash.pose_path(), default=str).encode())
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
