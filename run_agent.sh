#!/bin/bash
set -euo pipefail

# Default mode
MODE="gui"
PYTHON_PATH="/opt/conda/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python"
fi

# Helper to print usage
usage() {
    echo "Usage: $0 [option] [additional_args...]"
    echo "Options:"
    echo "  --nv, --headless   Run without vision/display mode (headless)."
    echo "  --gui, --vnc        Run with VNC/noVNC GUI mode enabled (default)."
    echo "  --vision            Run with vision mode enabled using host display."
    echo "  --no-lan            Disable integrated-server LAN publishing."
    echo "  --no-persistent     Full world reload on each episode reset."
    echo "  --lan-port PORT     Internal LAN bind port (default 25565)."
    echo "  -h, --help          Show this help message."
    exit 1
}

# Parse options
export MINERL_LAN_ENABLED="true"
export MINERL_PERSISTENT_SERVER="true"
export MINERL_LAN_PORT="25565"
AGENT_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nv|--headless)
            MODE="headless"
            shift
            ;;
        --gui|--vnc)
            MODE="gui"
            shift
            ;;
        --vision)
            MODE="vision"
            shift
            ;;
        --no-lan)
            export MINERL_LAN_ENABLED="false"
            shift
            ;;
        --no-persistent)
            export MINERL_PERSISTENT_SERVER="false"
            shift
            ;;
        --lan-port)
            export MINERL_LAN_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            AGENT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$MODE" = "headless" ]; then
    echo "Running without vision mode (headless)."
    exec xvfb-run -a "$PYTHON_PATH" agent.py "${AGENT_ARGS[@]}"

elif [ "$MODE" = "gui" ]; then
    echo "Running in GUI mode (VNC/noVNC enabled)."
    # Start noVNC websocket bridge
    websockify --web /usr/share/novnc 6080 localhost:5900 &
    
    # Run x11vnc and agent.py inside xvfb-run's display context
    exec xvfb-run -a -s "-screen 0 1980x1080x24" bash -c \
        'x11vnc -display "$DISPLAY" -nopw -listen 0.0.0.0 -forever -shared -q & exec "$0" "$@"' \
        "$PYTHON_PATH" agent.py "${AGENT_ARGS[@]}"

else
    echo "Running with vision mode enabled (using host display)."
    exec "$PYTHON_PATH" agent.py "${AGENT_ARGS[@]}"
fi