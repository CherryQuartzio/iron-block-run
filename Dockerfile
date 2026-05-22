FROM condaforge/miniforge3:24.11.3-0
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV_NAME=minerl
ENV TMPDIR=/opt/tmp
ENV TEMP=/opt/tmp
ENV TMP=/opt/tmp
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV JAVA_HOME=/opt/conda/envs/minerl
ENV PATH=/opt/conda/envs/minerl/bin:/opt/conda/bin:${PATH}
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
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/tmp /workspace
RUN conda create -y -n "${CONDA_ENV_NAME}" -c conda-forge \
    python=3.10 \
    pip \
    openjdk=8 \
    "setuptools<81" \
    wheel \
    && conda clean --all -f -y
RUN conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    python -m pip install --upgrade pip
RUN conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    python -m pip install --no-cache-dir git+https://github.com/minerllabs/minerl
RUN conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    python -m pip install --no-cache-dir stable-baselines3 opencv-python shimmy

# Create a non-root user for security
RUN useradd -ms /bin/bash minerluser
USER minerluser

# Switch to non-root user and set entrypoint
CMD ["bash"]