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

### Docker Swarm (VPS)

Deploy on the VPS Swarm cluster (not required for local dev compose):

```bash
docker stack deploy -c docker-compose.yml horserace
```

Open inbound TCP **25560** (game) and **6080** (noVNC) on the VPS firewall /
security group. Swarm maps external **25560** to internal **25565** where
MineRL binds the integrated server.

### Connect as a spectator

1. Start training inside the container: `./run_agent.sh`
2. Wait for the log line: `[LAN] Spectators can connect on internal port 25565`
3. In Minecraft 1.16.5: Multiplayer → Direct Connect → `YOUR_VPS_IP:25560`
4. Join in spectator mode; commands are enabled (`/tp`, `/gamemode`, etc.)
5. Spectators stay connected when the agent resets between episodes

### Local compose (non-Swarm)

```bash
docker compose build
docker compose up -d
docker compose exec minerl bash -c './run_agent.sh'
```

Direct Connect to `localhost:25560` when using the updated compose port mapping.

### Model compatibility

Checkpoints from `main` load unchanged via `load_agent.sh` / `./agent/` — no
changes to observation space, actions, or reward logic.

### Flags

- `./run_agent.sh --no-lan` — disable spectator port publishing
- `./run_agent.sh --no-persistent` — full world reload each episode (disconnects spectators)
- `./run_agent.sh --lan-port 25565` — change internal bind port

