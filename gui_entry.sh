#!/bin/bash
set -e

# Start noVNC websocket bridge
websockify --web /usr/share/novnc 6080 localhost:5900 &

# Inner script runs inside xvfb-run's display context, so x11vnc
# and agent.py share the exact same DISPLAY without polling.
cat > /tmp/run_with_vnc.sh << 'EOF'
#!/bin/bash
x11vnc -display "$DISPLAY" -nopw -listen 0.0.0.0 -forever -shared -q &
exec python agent.py
EOF
chmod +x /tmp/run_with_vnc.sh

exec xvfb-run -a -s "-screen 0 1980x1080x24" /tmp/run_with_vnc.sh
s