#!/bin/bash
# Select a saved agent checkpoint for agent.py to load on startup.
#
# Usage:
#   ./load_agent.sh saved_agents/saved_agent_20260601_160712.zip
#
# The checkpoint is a zip and SB3 loads zips directly, so this just places the
# chosen zip into ./agent/ (next to this script, where agent.py looks via
# LOAD_AGENT_DIR="agent"). No extraction — that would only force agent.py to
# re-zip it before loading. Any existing ./agent is replaced.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-agent-zip>" >&2
    exit 1
fi

SRC="$1"
if [[ ! -f "$SRC" ]]; then
    echo "Error: '$SRC' is not a file." >&2
    exit 1
fi

# Resolve to an absolute path before any directory juggling.
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"

# Sanity check: it should be a zip containing an SB3 model (a 'data' entry).
if ! unzip -l "$SRC" 2>/dev/null | grep -qw data; then
    echo "Error: '$SRC' does not look like an SB3 model zip (no 'data' entry)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/agent"

echo "Selecting agent:"
echo "  from: $SRC"
echo "  into: $TARGET/"

# Replace any existing ./agent and drop the zip in (no extraction).
rm -rf "$TARGET"
mkdir -p "$TARGET"
cp "$SRC" "$TARGET/$(basename "$SRC")"

echo "Done. ./agent holds $(basename "$SRC") — agent.py will load it on next launch."
