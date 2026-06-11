# Horace
*Formerly Iron Block Run*

A Minecraft reinforcement learning horce racer agent powered by [MineRL](https://minerl.readthedocs.io/en/latest/index.html)

## Setup

### Prerequisite

The bundled Docker Compose file assumed that you're running Debian Linux either from bare metal or with WSL2. You may made further tweaks to get it up and running with another operating system.

A machine with at least 8GB or RAM is recommended.

### Custom world

The agent will attempt to load the Minecraft `world` directory in the project root upon starting up. Otherwise, it will randomly generate a new world, which won't be useful for our use case. Drop your custom racetrack world into root using the same directory name.

### Running the agent

Clone this repo and build the Docker image. MineRL is a massive library that has the entire Minecraft 1.16.5 client bundled in, so building may take up to 15 minutes depending on processing power.

```bash
docker compose build
```

Once the image is built, run the respective compose file depending on what GPU your system has. Then exec into bash inside the container

```bash
docker compose up -d                             # If you have NVIDIA GPU
docker compose -f docker-compose-nogpu.yml up -d # CPU only

docker compose exec minerl bash
```

To begin running the agent, run the script inside the container

```bash
./run-agent.sh      # default with NoVNC
./run-agent.sh --nv # headless
```

### Load Checkpoint
After each run, the agent will save its parameter onto a zip file for restoring trained data in subsequent runs on the same machine or in another instance. To load a pre-trained agent on the same system, find the desired saved_agent zip in saved_agents/ and run:

```bash
bash load_agent.sh saved_agents/saved_agent_20260602_005514.zip
```

The agent now exists at agent/ and will be loaded at the start of training.

If you want to bring in an existing trained model from another instance, create the `agent` directory in root and drop the model zip into it.
