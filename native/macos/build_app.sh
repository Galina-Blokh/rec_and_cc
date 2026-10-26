#!/usr/bin/env bash
# Build dist/RecAndCC.app: a launcher bundle that runs the repo's Python as its child, so
# macOS privacy permissions (microphone, system audio) are attributed to "RecAndCC".
#
#   native/macos/build_app.sh             # -> <repo>/dist/RecAndCC.app
#   native/macos/build_app.sh --install   # ... and copy it to ~/Applications
#
# macOS remembers permissions by the app's code signature. The bundle is ad-hoc signed, so its
# identity is the hash of its contents: ANY change to the bundle makes it a "new" app and the
# permissions must be granted again. To keep that from happening on every update:
#   - the bundle holds only the launcher, Info.plist, launcher.json and the icon. Python code
#     runs from the repo and the rcc-audiotap helper lives in <repo>/bin (see native.py), so
#     neither needs an app rebuild;
#   - --install keeps the installed app untouched when the bundle contents are unchanged.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
pkg="$here/AudioTap"
app="$repo/dist/RecAndCC.app"
install=0
[[ "${1:-}" == "--install" ]] && install=1

"$here/build.sh"  # helper -> <repo>/bin/rcc-audiotap (outside the bundle, on purpose)
swift build --package-path "$pkg" -c release
bin_dir="$(swift build --package-path "$pkg" -c release --show-bin-path)"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp -f "$bin_dir/RecAndCC" "$app/Contents/MacOS/RecAndCC"

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

# Fingerprint of everything that ends up in the bundle (before signing), to detect no-op updates.
fingerprint="$(cd "$app/Contents" && cat MacOS/RecAndCC Info.plist Resources/launcher.json \
  $([[ -f Resources/AppIcon.icns ]] && echo Resources/AppIcon.icns) | shasum -a 256 | cut -d' ' -f1)"
echo "$fingerprint" > "$app/Contents/Resources/build-fingerprint"

codesign --force -s - "$app"
echo "Built $app"

if (( install )); then
  target="$HOME/Applications/RecAndCC.app"
  installed="$(cat "$target/Contents/Resources/build-fingerprint" 2>/dev/null || true)"
  if [[ "$installed" == "$fingerprint" ]]; then
    echo "Installed app is unchanged; kept $target as is (its permissions stay valid)."
  else
    mkdir -p "$HOME/Applications"
    rm -rf "$target"
    cp -R "$app" "$target"
    echo "Installed $target"
    if [[ -n "$installed" || -d "$HOME/Library/Logs/RecAndCC" ]]; then
      echo "NOTE: the app bundle changed, so macOS sees a new app. Restart RecAndCC and grant"
      echo "      Microphone + System audio again (menu bar → Permissions)."
    fi
  fi
fi
