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
if [ ! -e /workspace/pyproject.toml ]; then
    ln -s /pyproject.toml /workspace/pyproject.toml
fi

# Set git identity and trust the workspace (mounted volume may be owned by a different user)
git config --global user.email "agent@autoresearch"
git config --global user.name "Autoresearch Agent"
git config --global --add safe.directory /workspace

# Commit so git reset --hard never reverts the patches
if ! git diff --cached --quiet || ! git diff --quiet pyproject.toml || ! git diff --quiet uv.lock; then
    git add pyproject.toml uv.lock
    git commit -m "pyproject.toml and uv.lock changes"
fi

exec /bin/bash
