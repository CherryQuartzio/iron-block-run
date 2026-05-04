# Iron Block Run

A Minecraft horse racer agent powered by [MineRL](https://minerl.readthedocs.io/en/latest/index.html)

## Setup
---

### Prerequisite

The bundled Docker Compose file assumed that you're running Debian Linux either from bare metal or with WSL2. You may made further tweaks to get it up and running with another operating system.

A machine with at least 16GB or RAM is recommended.

### Run environment with Docker (recommended)

Clone this repo and run the following from the root. This will bring you into the containerized shell environment

```bash
docker compose build
docker compose up -d
docker compose exec minerl bash
conda activate minerl
```

### Manual install

1. Make sure conda is installed
2. Create the conda environment: `conda create -n minerl -c conda-forge python=3.10 pip openjdk=8`
3. Create a dedicated directory for this project and `cd` into it, then do the commands below:

```bash
conda activate minerl

# Free package/build caches
python -m pip cache purge
conda clean --all -y

# Use a temp directory on a filesystem with more free space
mkdir -p "$HOME/tmp" "$HOME/.cache/pip"
export TMPDIR="$HOME/tmp"
export TEMP="$HOME/tmp"
export TMP="$HOME/tmp"
export PIP_CACHE_DIR="$HOME/.cache/pip"

# install
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir git+https://github.com/minerllabs/minerl
```

## Run
---

To run the agent, you'll need to supply your own Minecraft 1.16.5 world and to edit the starting coordinate of the agent in the source file. It may take a while for the learning process.

```bash
# Start the agent
python agent.py
```