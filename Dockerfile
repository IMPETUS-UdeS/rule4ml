ARG GPU_TYPE=cuda

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base-cuda
FROM rocm/dev-ubuntu-22.04:6.4-complete AS base-rocm

FROM base-${GPU_TYPE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Redeclare GPU_TYPE for use in later stages
ARG GPU_TYPE=cuda
ARG REPO_URL
ARG REPO_REF=agent

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the repo from specified branch
RUN git clone --branch ${REPO_REF} ${REPO_URL} .

# Install uv and huggingface CLI
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv tool install huggingface_hub[cli]

# Download wa-hls4ml dataset
RUN mkdir -p /workspace/datasets/huggingface/wa-hls4ml/
RUN hf download --type dataset fastmachinelearning/wa-hls4ml \
    --local-dir /workspace/datasets/huggingface/wa-hls4ml/

# Install nodejs
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install agentic CLI tools (claude, opencode, gemini and codex)
RUN curl -fsSL https://claude.ai/install.sh | bash
RUN curl -fsSL https://opencode.ai/install | bash
RUN npm install -g @google/gemini-cli
RUN npm install -g @openai/codex

# Install python dependencies with the GPU-specific torch extra (cuda or rocm),
# move the venv, lock and datasets out, then wipe the workspace (re-cloned at entry)
RUN uv sync --extra ${GPU_TYPE} && \
    mv /workspace/.venv /venv && \
    mv /workspace/uv.lock /uv.lock && \
    mv /workspace/datasets /datasets && \
    rm -rf /workspace
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
# Never re-sync the pre-built venv: a sync without --extra would swap torch for the PyPI build
ENV UV_NO_SYNC=1

# Persist build args as env vars for the entrypoint
ENV REPO_URL=${REPO_URL}
ENV REPO_REF=${REPO_REF}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
