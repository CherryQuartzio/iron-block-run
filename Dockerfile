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
COPY patches/EnvServer.java /tmp/patches/EnvServer.java
RUN cp /tmp/patches/EnvServer.java \
    /opt/conda/lib/python3.10/site-packages/minerl/MCP-Reborn/src/main/java/com/minerl/multiagent/env/EnvServer.java \
    && cd /opt/conda/lib/python3.10/site-packages/minerl/MCP-Reborn \
    && ./gradlew shadowJar -x test
RUN python -m pip install --no-cache-dir stable-baselines3 opencv-python shimmy nbtlib

CMD ["bash"]