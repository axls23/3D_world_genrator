"""Find an Android phone camera for the streaming demo (demo/run_stream_demo.sh --phone).

Both routes use the USB-C cable (any USB-C / Thunderbolt port):

  uvc       Android 14+ "USB webcam" mode, where the maker kept it: the phone is a standard UVC
            camera. Recognised by the uvcvideo driver on an *external* USB port (sysfs
            removable != "fixed"; built-in laptop cameras sit on fixed ports) or an Android
            vendor id.
  android   any Android 12+ with USB debugging authorised: the camera is streamed over adb
            (android.py, scrcpy's device server; no app, no root, no sudo).

    python -m hypersplat.stream.phone detect    # JSON: {"mode", "device", "serial", "problems"}
    python -m hypersplat.stream.phone usb [back|front]
                                                # JSON: external USB camera (or phone) for the demo
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# USB vendor ids of Android phone makers (webcam mode keeps the phone's vendor id)
ANDROID_VENDORS = {
    "18d1": "Google", "04e8": "Samsung", "2717": "Xiaomi", "2a70": "OnePlus", "22b8": "Motorola",
    "0fce": "Sony", "1004": "LG", "12d1": "Huawei", "22d9": "OPPO/Realme", "2d95": "vivo",
    "0bb4": "HTC", "2e04": "HMD/Nokia", "19d2": "ZTE", "2ae5": "Fairphone",
}


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def list_cameras(sysfs: Path = Path("/sys/class/video4linux")) -> List[Dict]:
    """Capture nodes (index 0 of each device) with their USB / driver facts."""
    cams = []
    for node in sorted(sysfs.glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if _read(node / "index") not in ("", "0"):
            continue  # metadata / secondary nodes of the same camera
        dev = (node / "device").resolve()
        usb = dev.parent  # interface -> USB device
        driver = (dev / "driver").resolve().name if (dev / "driver").exists() else ""
        cams.append({
            "device": f"/dev/{node.name}",
            "name": _read(node / "name"),
            "driver": driver,
            "vendor": _read(usb / "idVendor"),
            "removable": _read(usb / "removable"),
            # platform device path on current v4l2loopback; older builds are virtual (no device
            # link), so fall back to its default / our card label
            "loopback": ("v4l2loopback" in str(dev) or "loopback" in driver
                         or _read(node / "name") in ("Dummy video device", "phone")),
        })
    return cams


def find_uvc_phone(cams: List[Dict]) -> Optional[Dict]:
    uvc = [c for c in cams if c["driver"] == "uvcvideo"]
    for c in uvc:  # an Android vendor id is the strongest signal
        if c["vendor"] in ANDROID_VENDORS:
            return c
    for c in uvc:  # otherwise any camera on an external port
        if c["removable"] and c["removable"] != "fixed":
            return c
    return None


# Built-in laptop cameras are UVC too; when the firmware does not mark their port "fixed"
# the name usually gives them away
BUILTIN_NAMES = re.compile(r"integrated|built-?in|internal|user facing|world facing|ir camera",
                           re.IGNORECASE)


def find_usb_camera(cams: List[Dict]) -> Optional[Dict]:
    """First UVC camera plugged into an external USB port (never the laptop's own camera)."""
    for c in cams:
        if c["driver"] != "uvcvideo" or c["removable"] == "fixed":
            continue
        if BUILTIN_NAMES.search(c["name"]) and c["vendor"] not in ANDROID_VENDORS:
            continue
        return c
    return None


def detect_usb(sysfs: Path = Path("/sys/class/video4linux"), facing: str = "back") -> Dict:
    """External USB camera, else an Android phone on the USB cable (webcam mode or USB
    debugging). ``problems`` says why a plugged-in phone is not usable yet."""
    cams = list_cameras(sysfs)
    listing = [f"{c['device']} ({c['name'] or '?'}, "
               f"{'built-in' if c['removable'] == 'fixed' else c['driver'] or '?'})" for c in cams]
    cam = find_usb_camera(cams)
    if cam:
        return {"device": cam["device"], "label": cam["name"], "cameras": listing, "problems": []}
    phone = detect(sysfs)
    if phone["mode"] == "android":
        return {"device": f"android://{phone['serial']}?camera={facing}", "label": phone["label"],
                "cameras": listing, "problems": []}
    # Only report phone problems when a phone is actually on the cable (not "adb missing")
    from . import android
    adb = android.adb_path()
    plugged = bool(adb and android.adb_devices(adb))
    return {"device": None, "label": None, "cameras": listing,
            "problems": phone["problems"] if plugged else []}


def find_loopback(cams: List[Dict]) -> Optional[Dict]:
    loops = [c for c in cams if c["loopback"]]
    named = [c for c in loops if "phone" in c["name"].lower()]
    return (named or loops or [None])[0]


def detect(sysfs: Path = Path("/sys/class/video4linux")) -> Dict:
    from . import android

    cams = list_cameras(sysfs)
    phone = find_uvc_phone(cams)
    if phone:
        maker = ANDROID_VENDORS.get(phone["vendor"], "external USB camera")
        return {"mode": "uvc", "device": phone["device"], "serial": None,
                "label": f"{phone['name']} ({maker}, USB webcam mode)", "problems": []}

    problems = []
    adb = android.adb_path()
    server, version = android.server_path()
    devs = android.adb_devices(adb) if adb else []
    ready = [d for d in devs if d["state"] == "device"]
    if not adb or not server or not version:
        problems.append("adb / scrcpy-server missing: unpack the scrcpy static build "
                        "(github.com/Genymobile/scrcpy releases) into ../tools/")
    elif not devs:
        problems.append("no phone seen by adb: enable Developer options > USB debugging, "
                        "set USB mode to File transfer")
    elif not ready:
        states = ", ".join(f"{d['serial']}={d['state']}" for d in devs)
        problems.append(f"phone not ready ({states}): unlock it and accept 'Allow USB debugging'")
    if problems:
        return {"mode": None, "device": None, "serial": None, "label": None, "problems": problems}
    serial = ready[0]["serial"]
    sdk, release = android.sdk_version(serial, adb)
    if sdk is not None and sdk < android.MIN_CAMERA_SDK:
        return {"mode": None, "device": None, "serial": None, "label": None, "problems": [
            f"phone {serial} runs Android {release or '?'} (API {sdk}): its camera can only be streamed "
            f"over adb from Android 12 on. Use an IP camera app instead, e.g. 'IP Webcam': start its "
            f"server, then `adb forward tcp:8085 tcp:8080` and run with source http://127.0.0.1:8085/video"]}
    return {"mode": "android", "device": f"android://{serial}", "serial": serial,
            "label": f"Android {serial} over adb (scrcpy server {version})", "problems": []}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("detect", "usb"):
        print(__doc__)
        sys.exit(2)
    facing = sys.argv[2] if len(sys.argv) > 2 else "back"
    print(json.dumps(detect() if cmd == "detect" else detect_usb(facing=facing)))


if __name__ == "__main__":
    main()
