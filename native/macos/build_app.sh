#!/usr/bin/env bash
# Build dist/RecAndCC.app: a launcher bundle that runs the repo's Python as its child, so
# macOS privacy permissions (microphone, system audio) are attributed to "RecAndCC".
#
#   native/macos/build_app.sh             # -> <repo>/dist/RecAndCC.app
#   native/macos/build_app.sh --install   # ... and copy it to ~/Applications
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
pkg="$here/AudioTap"
app="$repo/dist/RecAndCC.app"
install=0
[[ "${1:-}" == "--install" ]] && install=1

swift build --package-path "$pkg" -c release
bin_dir="$(swift build --package-path "$pkg" -c release --show-bin-path)"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp -f "$bin_dir/RecAndCC" "$app/Contents/MacOS/RecAndCC"
cp -f "$bin_dir/rcc-audiotap" "$app/Contents/MacOS/rcc-audiotap"

# Where the launcher finds Python. Baked in at build time; RCC_PYTHON overrides it at runtime.
python3 - "$repo" "$app/Contents/Resources/launcher.json" <<'PY'
import json, sys
repo, out = sys.argv[1], sys.argv[2]
json.dump({"repo": repo, "python": f"{repo}/.venv/bin/python", "args": ["-m", "recandcc", "run"]},
          open(out, "w"), indent=2)
PY

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleIdentifier</key>
	<string>com.gblokh.recandcc</string>
	<key>CFBundleName</key>
	<string>RecAndCC</string>
	<key>CFBundleDisplayName</key>
	<string>RecAndCC</string>
	<key>CFBundleExecutable</key>
	<string>RecAndCC</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleShortVersionString</key>
	<string>0.1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>14.4</string>
	<key>LSUIElement</key>
	<true/>
	<key>NSMicrophoneUsageDescription</key>
	<string>RecAndCC records your microphone during calls so your side of the conversation is transcribed too.</string>
	<key>NSAudioCaptureUsageDescription</key>
	<string>RecAndCC records the audio of your calls and meetings so it can transcribe them locally.</string>
</dict>
</plist>
PLIST

# Icon: render once, cache in .build (skipped silently if rendering fails).
icon_cache="$pkg/.build/AppIcon.icns"
if [[ ! -f "$icon_cache" ]]; then
  tmp="$(mktemp -d)"
  if swift "$here/make_icon.swift" "$tmp/icon.png" >/dev/null 2>&1; then
    iconset="$tmp/AppIcon.iconset"; mkdir -p "$iconset"
    for s in 16 32 128 256 512; do
      sips -z $s $s "$tmp/icon.png" --out "$iconset/icon_${s}x${s}.png" >/dev/null
      sips -z $((s*2)) $((s*2)) "$tmp/icon.png" --out "$iconset/icon_${s}x${s}@2x.png" >/dev/null
    done
    iconutil -c icns "$iconset" -o "$icon_cache" || true
  else
    echo "warning: icon rendering failed, building without an icon" >&2
  fi
  rm -rf "$tmp"
fi
[[ -f "$icon_cache" ]] && cp -f "$icon_cache" "$app/Contents/Resources/AppIcon.icns"

# Ad-hoc sign: the nested helper first (keeps its own identifier), then the bundle.
codesign -s - -f --identifier com.gblokh.recandcc.audiotap "$app/Contents/MacOS/rcc-audiotap"
codesign --force --deep -s - "$app"
echo "Built $app"

if (( install )); then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/RecAndCC.app"
  cp -R "$app" "$HOME/Applications/RecAndCC.app"
  echo "Installed $HOME/Applications/RecAndCC.app"
fi
