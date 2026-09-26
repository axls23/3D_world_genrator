"""Camera intrinsics: measure (metadata / 70-deg HFOV), then refine (ACE-Zero), then keep
reused cameras.bin files consistent with the refined focal."""

import struct

import pytest

from hypersplat.pipeline.params.strategies import discover, intrinsics


def _write_pinhole(path, w, h, f, model=1):
    params = [f, f, w / 2, h / 2] if model == 1 else [f, w / 2, h / 2]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<iiQQ", 1, model, w, h))
        fh.write(struct.pack(f"<{len(params)}d", *params))


# ----------------------------------------------------------------------------
# Initial focal
# ----------------------------------------------------------------------------
def test_heuristic_matches_old_70deg_hfov(drone_profile, whatsapp_profile):
    # The old hard-coded guesses these datasets were exported with
    d = intrinsics.initial_focal(drone_profile.get("frames.width"), drone_profile.get("frames.height"))
    assert d.strategy == "hfov-70"
    assert d.value == pytest.approx(drone_profile.get("_observed.heuristic_focal_px"), abs=0.5)
    w = intrinsics.initial_focal(whatsapp_profile.get("frames.width"), whatsapp_profile.get("frames.height"))
    assert w.value == pytest.approx(whatsapp_profile.get("_observed.heuristic_focal_px"), abs=0.01)


def test_parse_ffprobe_apple_35mm_tag():
    info = {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080,
                     "tags": {"handler_name": "Core Media Video"}},
                    {"codec_type": "audio", "tags": {}}],
        "format": {"tags": {"com.apple.quicktime.camera.focal_length.35mmEquivalent": "26",
                            "com.apple.quicktime.make": "Apple"}},
    }
    f35 = intrinsics.parse_focal_35mm(info)
    assert f35 == 26.0
    # 1920x1080 video extracted to 640x360 frames: 26/36 * 640
    d = intrinsics.initial_focal(640, 360, f35)
    assert d.strategy == "metadata-35mm"
    assert d.value == pytest.approx(462.22, abs=0.01)
    # Portrait frames use the long side too
    assert intrinsics.initial_focal(360, 640, f35).value == pytest.approx(462.22, abs=0.01)


def test_parse_ffprobe_exif_style_and_missing():
    info = {"streams": [{"codec_type": "video", "tags": {"FocalLengthIn35mmFormat": "24 mm"}}]}
    assert intrinsics.parse_focal_35mm(info) == 24.0
    # Drone demo: no focal tags at all -> None -> heuristic
    info = {"streams": [{"codec_type": "video", "tags": {"encoder": "AVC Coding"}}],
            "format": {"tags": {"major_brand": "mp42"}}}
    assert intrinsics.parse_focal_35mm(info) is None
    assert intrinsics.parse_focal_35mm({"format": {"tags": {"focal_length_35mm": "0"}}}) is None


def test_not_auto_applied():
    # Consumed directly by perception/manager, not a PipelineConfig attribute
    assert not any(p.NAME.startswith("FOCAL") for p in discover())


# ----------------------------------------------------------------------------
# Refined focal for the COLMAP export
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("fixture,expected", [("whatsapp_profile", 593.37), ("drone_profile", 317.08)])
def test_export_uses_ace_refined_focal(request, fixture, expected):
    profile = request.getfixturevalue(fixture)
    ace = profile.get("ace.focal_median")
    old = profile.get("_observed.heuristic_focal_px")
    initial = intrinsics.initial_focal(profile.get("frames.width"), profile.get("frames.height")).value
    # ACE-Zero writes (almost) the same refined focal for every frame
    focals = [ace] * profile.get("ace.n_registered")
    d = intrinsics.refined_focal(focals, initial)
    assert d.strategy == "ace-refined"
    assert d.value == pytest.approx(expected, abs=0.01)
    assert abs(d.value - old) / old > 0.2  # not the old guess
    assert intrinsics.refined_focal([], initial) is None


# ----------------------------------------------------------------------------
# Stale cameras.bin (--colmap-input / skip-ACE)
# ----------------------------------------------------------------------------
def test_stale_threshold():
    assert intrinsics.is_stale(342.76, 593.37)
    assert intrinsics.is_stale(457.0, 317.08)
    assert intrinsics.is_stale(317.08 * 1.025, 317.08)
    assert not intrinsics.is_stale(317.08 * 1.015, 317.08)
    assert not intrinsics.is_stale(593.0, 593.37)


@pytest.mark.parametrize("fixture", ["whatsapp_profile", "drone_profile"])
def test_stale_cameras_bin_rewritten(tmp_path, request, fixture):
    profile = request.getfixturevalue(fixture)
    w, h = profile.get("frames.width"), profile.get("frames.height")
    ace = profile.get("ace.focal_median")
    cams = tmp_path / "cameras.bin"
    _write_pinhole(cams, w, h, profile.get("_observed.heuristic_focal_px"))
    original = cams.read_bytes()

    d = intrinsics.fix_stale_cameras(cams, ace, w)
    assert d is not None and d.strategy == "stale-rewrite"
    (_, model, cw, ch, params), = intrinsics.read_colmap_cameras(cams)
    assert (model, cw, ch) == (1, w, h)
    assert params == pytest.approx([ace, ace, w / 2, h / 2])  # principal point untouched
    backup = tmp_path / "cameras.bin.bak"
    assert backup.read_bytes() == original

    # Idempotent; an existing backup is never overwritten
    assert intrinsics.fix_stale_cameras(cams, ace, w) is None
    _write_pinhole(cams, w, h, 100.0)
    assert intrinsics.fix_stale_cameras(cams, ace, w) is not None
    assert backup.read_bytes() == original


def test_close_focal_left_alone_and_simple_pinhole(tmp_path):
    cams = tmp_path / "cameras.bin"
    _write_pinhole(cams, 640, 360, 317.08 * 1.01)
    assert intrinsics.fix_stale_cameras(cams, 317.08, 640) is None
    assert not (tmp_path / "cameras.bin.bak").exists()

    _write_pinhole(cams, 640, 360, 457.0, model=0)
    assert intrinsics.fix_stale_cameras(cams, 317.08, 640) is not None
    (_, model, _, _, params), = intrinsics.read_colmap_cameras(cams)
    assert model == 0 and params == pytest.approx([317.08, 320.0, 180.0])


def test_ace_focal_rescaled_to_camera_width(tmp_path):
    # cameras.bin describing images at 2x the resolution ACE-Zero ran on
    cams = tmp_path / "cameras.bin"
    _write_pinhole(cams, 1280, 720, 634.17)
    assert intrinsics.fix_stale_cameras(cams, 317.08, 640) is None
