#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

ENV_NAME="${ENV_NAME:-soft_gac}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

# Function to print messages with a timestamp.
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

# Ensure that conda is discoverable even in a fresh non-interactive shell.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BASE="$("$HOME/miniconda3/bin/conda" info --base)"
    export PATH="$CONDA_BASE/bin:$PATH"
else
    log "Could not find conda; make sure Miniconda/Conda is installed."
    exit 1
fi

# Ensure that the conda base environment is initialized for this shell.
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    log "Could not find conda.sh under $CONDA_BASE; make sure conda is initialized."
    exit 1
fi

# Create the conda environment only if it does not already exist.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' already exists; reusing it."
else
    log "Creating the conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

log "Activating the '$ENV_NAME' environment..."
conda activate "$ENV_NAME"

log "Upgrading installer tooling..."
python -m pip install --upgrade pip 'setuptools<81' wheel

if [ -f "$REQ_FILE" ]; then
    log "Installing project requirements from $REQ_FILE..."
    python -m pip install --prefer-binary -r "$REQ_FILE"
else
    log "$REQ_FILE not found. Skipping project requirements install."
fi

log "Verifying key imports..."
python - <<'PY'
import importlib
for name in ["torch", "jax", "flax", "optax", "gymnasium", "mujoco"]:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'import ok')}")
PY

log "Setup complete! The '$ENV_NAME' environment is ready for use."
