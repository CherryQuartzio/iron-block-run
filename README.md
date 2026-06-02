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