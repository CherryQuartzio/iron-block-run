# Iron Block Run

A Minecraft horse racer agent powered by [MineRL](https://minerl.readthedocs.io/en/latest/index.html)

## Setup
---

### Prerequisite

The bundled Docker Compose file assumed that you're running Debian Linux either from bare metal or with WSL2. You may made further tweaks to get it up and running with another operating system.

A machine with at least 16GB or RAM is recommended.

### Run environment with Docker

Clone this repo and run the following from the root. This will bring you into the containerized shell environment

```bash
docker compose build
docker compose up -d
docker compose exec minerl bash
```

### Run environment using Conda

Clone this repo and run the following from the root.

```bash
# setup the environment
bash setup_conda.sh
```

## Run
---

To run the agent, you'll need to supply your own Minecraft 1.16.5 world and to edit the starting coordinate of the agent in the source file. It may take a while for the learning process.

```bash
# Start the agent
python agent.py
```

## Load Checkpoint
To load a pre-trained agent, find the desired saved_agent zip in saved_agents/ and run:

```bash
bash load_agent.sh saved_agents/saved_agent_20260602_005514.zip
```

The agent now exists at agent/ and will be loaded at the start of training.

## Spectating (multiplayer branch)

This branch keeps MineRL's integrated server online across episode resets and
publishes it for vanilla **Minecraft 1.16.5 Java Edition** spectators.

### Docker deployment

The image is pinned to `linux/amd64` because MineRL ships x86-64 LWJGL natives.
On ARM hosts (e.g. Oracle Ampere), Docker runs the container under QEMU emulation.

**ARM build prerequisite:** cross-platform builds need QEMU binfmt on the host.
Without it, `docker compose build` fails at an early `RUN apt-get ...` step with
`exec /bin/sh: exec format error`. Run this once before building:

```bash
bash scripts/setup-docker-amd64-on-arm.sh
```

**Local or remote (standalone Docker):**

```bash
docker compose build
docker compose up -d
docker compose exec minerl bash -c './run_agent.sh'
```

On a remote server, copy the project, run the same commands, and open inbound
TCP **25565** (Minecraft LAN) and **6080** (noVNC) on the host firewall.

For a one-off run without compose:

```bash
docker run --platform linux/amd64 -it -v $(pwd):/workspace -p 25565:25565 -p 6080:6080 minerl-dev bash
```

### Connect as a spectator

1. Start training inside the container: `./run_agent.sh`
2. Wait for the log line: `[LAN] Spectators can connect on port 25565`
3. In Minecraft 1.16.5: Multiplayer → Direct Connect → `HOST:25565`
   - Local: `localhost:25565`
   - Remote server: `YOUR_SERVER_IP:25565`
4. Join in spectator mode; commands are enabled (`/tp`, `/gamemode`, etc.)
5. Spectators stay connected when the agent resets between episodes

### Model compatibility

Checkpoints from `main` load unchanged via `load_agent.sh` / `./agent/` — no
changes to observation space, actions, or reward logic.

### Flags

- `./run_agent.sh --no-lan` — disable spectator port publishing
- `./run_agent.sh --no-persistent` — full world reload each episode (disconnects spectators)
- `./run_agent.sh --lan-port 25565` — change internal bind port

### Steering regression validation (server)

After rebuilding the image on the server, compare `main` vs `lan` with the same
checkpoint in `./agent/`. Evaluation logs now include per-episode `env_fps`
(Python `env.step()` throughput) and `Step 0` pose lines.

| Run | Command | What to check |
|-----|---------|---------------|
| Baseline | `main` branch, `python agent.py` (eval) | Lap complete; note `env_fps` and Step 0 pose (~`Z=-152` after the mount walk) |
| LAN + spectator | `lan` branch, `./run_agent.sh` | Mounted at a Step 0 pose matching main (~`Z=-152`); no `moved too quickly` at start; reaches CP_B |
| LAN, no spectator | `./run_agent.sh --no-lan` | Same as above; compare `env_fps` to isolate server load |

Note: `Y` varies with track elevation. The `mount may have failed` warning can
be a false positive when `ypos` barely changes on sloped track. More reliable:
check Step 0 `Pitch` — it must be `0°`, not `20°` (a pitched-down camera means
the mount look-up step was skipped and the policy will veer immediately).

Server tick load is recorded in the PlayRecorder jsonl as `serverTickDurationMs`
(under the MCP-Reborn run logs directory). Values consistently above ~50 ms
indicate sub-20-TPS server pressure (often when a spectator is connected).

Look for `[HorseAI] Restored AI on mounted horse (NoAI cleared)` in the
Minecraft log on each episode mount.

