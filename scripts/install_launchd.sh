#!/usr/bin/env bash
# Install the capture LaunchAgent and the 06:15 loss-check timer for this user (no sudo, no
# system settings). Usage: scripts/install_launchd.sh [capture|loss|all] [--dry-run]
# Starting the capture agent starts a second consumer in the same Kafka group, so stop any
# running capture first (pkill -INT -f scripts/capture_darwin.py).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
WHAT="${1:-all}"
DRY="${2:-}"
UV="$(command -v uv)"
PATHVAL="$(dirname "$UV"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
DEST="$HOME/Library/LaunchAgents"
mkdir -p logs "$DEST"

install_one() {
  local name="$1" src="scripts/launchd/$1.plist.in" out="$DEST/$1.plist"
  [[ "$DRY" == "--dry-run" ]] && out="${TMPDIR:-/tmp}/$1.plist"
  sed -e "s|@REPO@|$REPO|g" -e "s|@UV@|$UV|g" -e "s|@PATH@|$PATHVAL|g" "$src" > "$out"
  plutil -lint "$out" > /dev/null
  [[ "$DRY" == "--dry-run" ]] && { echo "rendered $out"; return; }
  launchctl bootout "gui/$(id -u)/$name" 2> /dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$out"
  echo "loaded $name"
}

[[ "$WHAT" == capture || "$WHAT" == all ]] && install_one com.railrisk.capture
[[ "$WHAT" == loss || "$WHAT" == all ]] && install_one com.railrisk.capture-loss
exit 0
