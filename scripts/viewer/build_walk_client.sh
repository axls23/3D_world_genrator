#!/usr/bin/env bash
# Build viser's web client with the hypersplat walk-mode patch and install it into the venv.
#
#   scripts/viewer/build_walk_client.sh            build + install (idempotent)
#   scripts/viewer/build_walk_client.sh --restore  put the stock client back
#
# The patch (patches/viser_walk_client.patch) adds SetWalkBoundsMessage and a ground-level walk
# mode to CameraControls.tsx; hypersplat/stream/walk_bounds.py sends the bounds. The stock
# build is kept as client/build.orig. A marker file in client/build tells the trainer the walk
# client is installed. Needs node >= 20 (npm). Pinned to viser 1.1.1: the patch targets that
# client source.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$REPO/.venv_3dgrut/bin/python}"
PATCH="$REPO/patches/viser_walk_client.patch"
WORK="${HYPERSPLAT_CACHE:-$HOME/.cache/hypersplat}/viser_walk_client"
MARKER=".hypersplat_walk"
WANT_VISER="1.1.1"

VISER_DIR="$("$PY" -c 'import os, viser; print(os.path.dirname(viser.__file__))')"
VISER_VER="$("$PY" -c 'import viser; print(viser.__version__)')"
CLIENT="$VISER_DIR/client"

if [[ "${1:-}" == "--restore" ]]; then
  if [[ -d "$CLIENT/build.orig" ]]; then
    rm -rf "$CLIENT/build"
    mv "$CLIENT/build.orig" "$CLIENT/build"
    echo "Restored stock viser client."
  else
    echo "No build.orig to restore."
  fi
  exit 0
fi

if [[ "$VISER_VER" != "$WANT_VISER" ]]; then
  echo "viser $VISER_VER installed; the walk patch targets $WANT_VISER. Refusing." >&2
  exit 1
fi
command -v npm >/dev/null || { echo "npm not found (node >= 20 needed)" >&2; exit 1; }

PATCH_SHA="$(sha256sum "$PATCH" | cut -d' ' -f1)"
if [[ -f "$CLIENT/build/$MARKER" && "$(cat "$CLIENT/build/$MARKER")" == "$PATCH_SHA" ]]; then
  echo "Walk client already installed ($CLIENT/build)."
  exit 0
fi

# Mirror the package layout: client/src imports ../../_assets
rm -rf "$WORK/viser/client.new"
mkdir -p "$WORK/viser"
ln -sfn "$VISER_DIR/_assets" "$WORK/viser/_assets"
(cd "$CLIENT" && tar --exclude=build --exclude=build.orig --exclude=node_modules -cf - .) \
  | (mkdir -p "$WORK/viser/client.new" && tar -xf - -C "$WORK/viser/client.new")
# Reuse node_modules from a previous build (same package-lock)
if [[ -d "$WORK/viser/client/node_modules" ]] && \
   cmp -s "$WORK/viser/client/package-lock.json" "$WORK/viser/client.new/package-lock.json"; then
  mv "$WORK/viser/client/node_modules" "$WORK/viser/client.new/"
fi
rm -rf "$WORK/viser/client"
mv "$WORK/viser/client.new" "$WORK/viser/client"

cd "$WORK/viser"
patch -p1 --forward < "$PATCH"
cd client
[[ -d node_modules ]] || npm ci --no-audit --no-fund
npm run build

if [[ ! -d "$CLIENT/build.orig" ]]; then
  cp -a "$CLIENT/build" "$CLIENT/build.orig"
fi
rm -rf "$CLIENT/build"
cp -a build "$CLIENT/build"
echo "$PATCH_SHA" > "$CLIENT/build/$MARKER"
echo "Installed walk client into $CLIENT/build (stock build kept in build.orig)."
