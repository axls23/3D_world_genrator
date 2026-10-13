"""Find IP cameras / phone camera apps on the local network (demo/run_stream_demo.sh --wifi).

Scans the /24 of every private IPv4 interface for the ports camera apps use, then confirms each
open port by asking for a video stream: HTTP paths that answer with multipart/MJPEG or video
content, and RTSP DESCRIBE. Only private addresses are scanned; nothing is sent beyond one
connect + one short request per candidate.

    python -m hypersplat.stream.wifi [--timeout S] [--subnet 192.168.1.0/24]
        JSON list of {"url", "kind", "host", "port"}, best first
"""

import argparse
import ipaddress
import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# IP Webcam 8080, DroidCam 4747, common custom 8070/8081, RTSP 554/8554, RTMP-less cameras 8000
HTTP_PORTS = (8080, 8070, 8081, 4747, 8000, 8888, 80)
RTSP_PORTS = (554, 8554)
HTTP_PATHS = ("/video", "/videofeed", "/mjpegfeed", "/mjpeg", "/stream", "/video?640x480", "/h264_ulaw.sdp")
RTSP_PATHS = ("/", "/live", "/stream1", "/h264", "/cam")


def local_subnets() -> List[ipaddress.IPv4Network]:
    nets = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                iface = ipaddress.ip_interface(parts[parts.index("inet") + 1])
                if iface.ip.is_private and not iface.ip.is_link_local and iface.network.prefixlen >= 16:
                    # never sweep more than a /24 around us
                    nets.append(ipaddress.ip_network(f"{iface.ip}/{max(iface.network.prefixlen, 24)}", strict=False))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return nets


def _open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(host: str, port: int, path: str, timeout: float) -> Optional[str]:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(f"GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: hypersplat\r\n\r\n".encode())
            head = s.recv(1024).decode("latin-1", "replace").lower()
    except OSError:
        return None
    if not head.startswith("http/1.") or " 200 " not in head.split("\r\n", 1)[0]:
        return None
    if "multipart/x-mixed-replace" in head or "video/" in head or "image/jpeg" in head and "boundary" in head:
        return f"http://{host}:{port}{path}"
    return None


def _rtsp_probe(host: str, port: int, path: str, timeout: float) -> Optional[str]:
    url = f"rtsp://{host}:{port}{path}"
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n".encode())
            head = s.recv(512).decode("latin-1", "replace")
    except OSError:
        return None
    first = head.split("\r\n", 1)[0]
    if first.startswith("RTSP/1.0 200"):
        return url
    if first.startswith("RTSP/1.0 401"):  # camera is there but wants a login: report host-only
        return url + "  (needs rtsp://user:pass@...)"
    return None


def _check(host: str, port: int, timeout: float) -> Optional[Dict]:
    if not _open(host, port, timeout):
        return None
    if port in RTSP_PORTS:
        for p in RTSP_PATHS:
            url = _rtsp_probe(host, port, p, timeout * 2)
            if url:
                return {"url": url, "kind": "rtsp", "host": host, "port": port}
        return None
    for p in HTTP_PATHS:
        url = _http_probe(host, port, p, timeout * 2)
        if url:
            return {"url": url, "kind": "http", "host": host, "port": port}
    return None


def discover(subnets: Optional[List[ipaddress.IPv4Network]] = None, timeout: float = 0.4) -> List[Dict]:
    nets = subnets if subnets is not None else local_subnets()
    me = set()
    try:
        if subnets is not None:  # explicit subnet: scan everything in it
            raise OSError
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=5).stdout
        me = {l.split()[3].split("/")[0] for l in out.splitlines() if "inet" in l}
    except (OSError, subprocess.SubprocessError):
        pass
    jobs = [(str(h), p) for n in nets for h in n.hosts() if str(h) not in me
            for p in HTTP_PORTS + RTSP_PORTS]
    with ThreadPoolExecutor(max_workers=256) as pool:
        found = [r for r in pool.map(lambda j: _check(j[0], j[1], timeout), jobs) if r]
    found.sort(key=lambda r: (r["kind"] != "http", r["host"], r["port"]))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timeout", type=float, default=0.4)
    ap.add_argument("--subnet", action="append", help="CIDR to scan (default: local /24s)")
    a = ap.parse_args()
    try:
        nets = [ipaddress.ip_network(s, strict=False) for s in a.subnet] if a.subnet else None
    except ValueError as e:
        sys.exit(f"bad --subnet: {e}")
    if nets is None and not local_subnets():
        print("[]")
        sys.stderr.write("wifi: no private IPv4 network found (is the PC on Wi-Fi/LAN?)\n")
        return
    print(json.dumps(discover(nets, a.timeout)))


if __name__ == "__main__":
    main()
