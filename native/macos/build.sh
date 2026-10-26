#!/usr/bin/env bash
# Build the rcc-audiotap helper and install it to <repo>/bin/rcc-audiotap.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
pkg="$here/AudioTap"

swift build --package-path "$pkg" -c release
bin_dir="$(swift build --package-path "$pkg" -c release --show-bin-path)"

mkdir -p "$repo/bin"
cp -f "$bin_dir/rcc-audiotap" "$repo/bin/rcc-audiotap"
# Ad-hoc signature: required on Apple silicon, and gives TCC a stable code identity.
codesign -s - -f --identifier com.gblokh.recandcc.audiotap "$repo/bin/rcc-audiotap"

echo "Installed $repo/bin/rcc-audiotap"
