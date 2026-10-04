"""Android phone camera over the USB cable, without root, sudo or a virtual camera device.

    android://                    first adb device, back camera
    android://SERIAL?camera=front&max_size=1920&fps=30

Uses scrcpy's device-side server (the official static build ships it together with adb):
adb pushes scrcpy-server to the phone, forwards a local TCP port to its socket, and starts it
with ``video_source=camera raw_stream=true``, so the phone sends a plain H.264 elementary
stream that OpenCV/FFmpeg decode like any network camera. Needs USB debugging enabled and
authorised on the phone (Android 12+ for camera capture).

Tools are looked up in $HYPERSPLAT_ADB / $HYPERSPLAT_SCRCPY_SERVER, then the scrcpy bundle in
<repo>/../tools/scrcpy-linux-*/, then PATH.
"""

import logging
import os
import random
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
DEVICE_SERVER_PATH = "/data/local/tmp/hypersplat-scrcpy-server.jar"


def _bundle() -> Optional[Path]:
    dirs = sorted((d for d in TOOLS_DIR.glob("scrcpy-linux-*") if d.is_dir()), reverse=True)
    return dirs[0] if dirs else None


def adb_path() -> Optional[str]:
    env = os.environ.get("HYPERSPLAT_ADB")
    if env:
        return env
    b = _bundle()
    if b and (b / "adb").exists():
        return str(b / "adb")
    return shutil.which("adb")


def server_path() -> Tuple[Optional[str], Optional[str]]:
    """(scrcpy-server path, its version) - the server must be started with its exact version."""
    env = os.environ.get("HYPERSPLAT_SCRCPY_SERVER")
    b = _bundle()
    path = env or (str(b / "scrcpy-server") if b and (b / "scrcpy-server").exists() else None)
    if path is None:
        return None, None
    version = os.environ.get("HYPERSPLAT_SCRCPY_VERSION")
    if not version:
        m = re.search(r"v?(\d+\.\d+(?:\.\d+)?)", Path(path).parent.name if not env else Path(path).name)
        version = m.group(1) if m else None
    return path, version


def adb_devices(adb: Optional[str] = None) -> list:
    adb = adb or adb_path()
    if not adb:
        return []
    out = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=20).stdout
    return [{"serial": p[0], "state": p[1]} for p in (l.split() for l in out.splitlines()[1:]) if len(p) >= 2]


def parse_android_spec(spec: str) -> Dict:
    u = urlparse(spec)
    q = {k: v[-1] for k, v in parse_qs(u.query).items()}
    return {"serial": u.netloc or None, "camera": q.get("camera", "back"),
            "max_size": int(q.get("max_size", 1920)), "fps": int(q.get("fps", 30))}


def server_args(version: str, scid: str, camera: str, max_size: int, fps: int) -> list:
    """Arguments of com.genymobile.scrcpy.Server for a raw H.264 camera stream."""
    return [version, f"scid={scid}", "log_level=warn", "tunnel_forward=true", "video=true",
            "audio=false", "control=false", "video_source=camera", f"camera_facing={camera}",
            f"camera_fps={fps}", f"max_size={max_size}", "video_codec=h264", "raw_stream=true",
            "send_dummy_byte=false", "cleanup=true"]


class AndroidCamera:
    """Starts the camera stream on the phone; ``url`` is what OpenCV reads."""

    def __init__(self, spec: str):
        self.opts = parse_android_spec(spec)
        self.adb = adb_path()
        self.server, self.version = server_path()
        if not self.adb:
            raise RuntimeError("adb not found (install android-tools or unpack scrcpy into tools/)")
        if not self.server or not self.version:
            raise RuntimeError("scrcpy-server not found (unpack the scrcpy static build into tools/)")
        ready = [d["serial"] for d in adb_devices(self.adb) if d["state"] == "device"]
        serial = self.opts["serial"] or (ready[0] if ready else None)
        if serial is None or serial not in ready:
            states = ", ".join(f"{d['serial']}={d['state']}" for d in adb_devices(self.adb)) or "none"
            raise RuntimeError(f"no authorised adb device (seen: {states}); enable USB debugging "
                               "on the phone and accept the prompt")
        self.serial = serial
        self.scid = f"{random.randrange(1 << 31):08x}"
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None

    def device_state(self) -> str:
        """adb state of this phone: device | unauthorized | offline | ... | missing."""
        try:
            return next((d["state"] for d in adb_devices(self.adb) if d["serial"] == self.serial), "missing")
        except (OSError, subprocess.SubprocessError):
            return "missing"

    def restart(self) -> str:
        """Start a fresh camera stream after the phone dropped off USB (new socket name)."""
        self.close()
        self.scid = f"{random.randrange(1 << 31):08x}"
        return self.start()

    def _adb(self, *args, **kw):
        return subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True, text=True,
                              timeout=kw.pop("timeout", 30), **kw)

    def start(self, wait: float = 15.0) -> str:
        r = self._adb("push", self.server, DEVICE_SERVER_PATH)
        if r.returncode != 0:
            raise RuntimeError(f"adb push failed: {r.stderr.strip()}")
        r = self._adb("forward", "tcp:0", f"localabstract:scrcpy_{self.scid}")
        if r.returncode != 0 or not r.stdout.strip().isdigit():
            raise RuntimeError(f"adb forward failed: {r.stderr.strip() or r.stdout.strip()}")
        self.port = int(r.stdout.strip())
        o = self.opts
        cmd = [self.adb, "-s", self.serial, "shell", f"CLASSPATH={DEVICE_SERVER_PATH}", "app_process", "/",
               "com.genymobile.scrcpy.Server",
               *server_args(self.version, self.scid, o["camera"], o["max_size"], o["fps"])]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # The server listens once it has opened the camera; wait until the socket accepts
        end = time.time() + wait
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"scrcpy server exited: {self.proc.stdout.read().strip()[-500:]}")
            if self._listening():
                logger.info(f"[Android] {self.serial} {o['camera']} camera streaming on tcp:{self.port}")
                return self.url
            time.sleep(0.3)
        self.close()
        raise RuntimeError("scrcpy server did not start listening (camera busy or unsupported?)")

    def _listening(self) -> bool:
        # With tunnel_forward adb accepts locally at once; the server side is ready when the
        # device socket exists
        r = self._adb("shell", f"cat /proc/net/unix | grep -q scrcpy_{self.scid}", timeout=10)
        return r.returncode == 0

    @property
    def url(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.port is not None:
            self._adb("forward", "--remove", f"tcp:{self.port}", timeout=10)
            self.port = None
