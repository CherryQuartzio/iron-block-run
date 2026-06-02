FROM condaforge/miniforge3:24.11.3-0
ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/opt/tmp
ENV TEMP=/opt/tmp
ENV TMP=/opt/tmp
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV JAVA_HOME=/opt/conda
ENV PATH=/opt/conda/bin:${PATH}
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/tmp /workspace
RUN conda install -y -c conda-forge \
    python=3.10 \
    pip \
    openjdk=8 \
    "setuptools<81" \
    wheel \
    && conda clean --all -f -y
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir git+https://github.com/minerllabs/minerl

# --- Patch-independent layers --------------------------------------------
# Everything that does NOT depend on patches/* lives above the COPY below, so
# editing a patch never invalidates these (potentially slow) install layers.
RUN python -m pip install --no-cache-dir stable-baselines3 opencv-python shimmy nbtlib tensorboard
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    x11vnc \
    novnc \
    websockify \
    xdotool \
    openbox

# --- Patch layer (rebuilds when patches/* changes) -----------------------
# Keep this LAST: changing patches/EnvServer.java invalidates only the COPY
# and the gradle recompile below — nothing above is re-run. The recompile of
# the MCP-Reborn jar is the unavoidable cost of a patch change.
COPY patches/EnvServer.java /tmp/patches/EnvServer.java
RUN cp /tmp/patches/EnvServer.java \
    /opt/conda/lib/python3.10/site-packages/minerl/MCP-Reborn/src/main/java/com/minerl/multiagent/env/EnvServer.java \
    && cd /opt/conda/lib/python3.10/site-packages/minerl/MCP-Reborn \
    && ./gradlew shadowJar -x test
CMD ["bash"]
