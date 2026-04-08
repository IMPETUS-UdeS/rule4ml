#!/bin/bash
set -e

# Clone the repo into the mounted workspace if it's empty
if [ -z "$(ls -A /workspace)" ]; then
    echo "Workspace is empty, cloning repo..."
    git clone --branch "${REPO_REF}" "${REPO_URL}" /workspace
fi

cd /workspace

# Link the pre-built venv, uv.lock and datasets folder 
if [ ! -e /workspace/.venv ]; then
    ln -s /venv /workspace/.venv
fi
if [ ! -e /workspace/uv.lock ]; then
    ln -s /uv.lock /workspace/uv.lock
fi
if [ ! -e /workspace/datasets ]; then
    ln -s /datasets /workspace/datasets
fi

# Re-apply GPU-specific torch sources so uv run uses the pre-built venv
if [ "$GPU_TYPE" = "cuda" ]; then
    uv add --index pytorch-cuda=https://download.pytorch.org/whl/cu128 torch; \
elif [ "$GPU_TYPE" = "rocm" ]; then
    uv add --index pytorch-rocm=https://download.pytorch.org/whl/rocm7.2 torch pytorch-triton-rocm; \
fi

# Set git identity and trust the workspace
git config --global user.email "agent@autoresearch"
git config --global user.name "Autoresearch Agent"
git config --global --add safe.directory /workspace

# Commit so git reset --hard never reverts the patches
if ! git diff --cached --quiet || ! git diff --quiet pyproject.toml; then
    git add pyproject.toml
    git commit -m "pyproject.toml GPU-specific patches"
fi

exec /bin/bash
