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
    git curl ca-certificates \
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

# Install agentic CLI tools (claude, codex, gemini)
RUN npm install -g @anthropic-ai/claude-code
RUN npm install -g @google/gemini-cli
RUN npm install -g @openai/codex

# Add GPU-specific torch sources to pyproject.toml
RUN if [ "$GPU_TYPE" = "cuda" ]; then \
        uv add --index pytorch-cuda=https://download.pytorch.org/whl/cu128 torch; \
    elif [ "$GPU_TYPE" = "rocm" ]; then \
        uv add --index pytorch-rocm=https://download.pytorch.org/whl/rocm7.2 torch pytorch-triton-rocm; \
    fi

# Install python dependencies, move files, then wipe the workspace (re-cloned at entry)
RUN uv sync && \
    mv /workspace/.venv /venv && \
    mv /workspace/uv.lock /uv.lock && \
    mv /workspace/datasets /datasets && \
    mv /workspace/pyproject.toml /pyproject.toml && \
    rm -rf /workspace
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Persist build args as env vars for the entrypoint
ENV REPO_URL=${REPO_URL}
ENV REPO_REF=${REPO_REF}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
