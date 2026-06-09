#!/bin/bash
# Enable building and running linux/amd64 images on ARM64 hosts (Oracle Ampere, etc.).
# MineRL requires amd64 because MCP-Reborn bundles x86-64 LWJGL natives.
#
# Run once on the VPS before: docker compose build
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
    echo "Host is $(uname -m); QEMU binfmt registration is only needed on ARM64."
    exit 0
fi

echo "Registering QEMU user emulation for foreign Docker platforms (amd64)..."
docker run --rm --privileged tonistiigi/binfmt --install amd64

echo "Verifying amd64 binfmt handler..."
if ! grep -q 'qemu' /proc/sys/fs/binfmt_misc/status 2>/dev/null; then
    echo "Warning: binfmt may not be active. Try: sudo systemctl restart systemd-binfmt.service"
fi

echo "Done. You can now run: docker compose -f docker-compose-nogpu.yaml build"
