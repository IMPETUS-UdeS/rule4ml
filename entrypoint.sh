#!/bin/bash
set -e

# Clone the repo into the mounted workspace if it's empty
if [ -z "$(ls -A /workspace)" ]; then
    echo "Workspace is empty, cloning repo..."
    git clone --branch "${REPO_REF}" "${REPO_URL}" /workspace
fi

cd /workspace

# Link the pre-built venv, uv.lock and pyproject.toml 
if [ ! -e /workspace/.venv ]; then
    ln -s /venv /workspace/.venv
fi
if [ ! -e /workspace/uv.lock ]; then
    ln -s /uv.lock /workspace/uv.lock
fi

# Re-apply GPU-specific torch sources so uv run uses the pre-built venv
if ! grep -q "tool.uv.sources" pyproject.toml; then
    if [ "$GPU_TYPE" = "cuda" ]; then
        printf '\n[[tool.uv.index]]\nname = "pytorch-cuda"\nurl = "https://download.pytorch.org/whl/cu128"\nexplicit = true\n\n[tool.uv.sources]\ntorch = { index = "pytorch-cuda" }\n' >> pyproject.toml; \
    elif [ "$GPU_TYPE" = "rocm" ]; then
        printf '\n[[tool.uv.index]]\nname = "pytorch-rocm"\nurl = "https://download.pytorch.org/whl/rocm7.2"\n\n[tool.uv.sources]\ntorch = { index = "pytorch-rocm" }\npytorch-triton-rocm = { index = "pytorch-rocm" }\n' >> pyproject.toml; \
    fi
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
