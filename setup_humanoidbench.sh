#!/bin/bash
set -e

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

install_system_gl_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; skipping system OpenGL/EGL package install."
        return
    fi

    local sudo_cmd=()
    if [ "$(id -u)" -ne 0 ]; then
        if sudo -n true >/dev/null 2>&1; then
            sudo_cmd=(sudo)
        else
            log "Non-interactive sudo unavailable; skipping system OpenGL/EGL package install."
            return
        fi
    fi

    log "Installing system OpenGL/EGL packages for headless HumanoidBench..."
    "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -y
    "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y \
        libegl-mesa0 \
        libegl1 \
        libgbm1 \
        libgl1 \
        libgl1-mesa-dri \
        libgles2 \
        libglfw3 \
        libglvnd0 \
        libglx0 \
        libopengl0 \
        libosmesa6 \
        libx11-xcb1
}

if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    log "Could not find conda.sh; make sure conda is installed and initialized."
    exit 1
fi

install_system_gl_deps

ENV_NAME="${ENV_NAME:-soft_gac_humanoidbench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' already exists."
else
    log "Creating the conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

log "Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"

log "Installing SoftGAC requirements..."
pip install --upgrade pip 'setuptools<81' wheel
pip install --prefer-binary -r requirements.txt

log "Configuring dedicated HumanoidBench compatibility packages..."
pip uninstall -y shimmy || true
pip install --no-deps mujoco==3.1.6

log "Installing HumanoidBench as editable source without dependency resolution..."
export PIP_SRC="${PIP_SRC:-$HOME/src}"
mkdir -p "$PIP_SRC"
pip install --no-deps -e "git+https://github.com/carlosferrazza/humanoid-bench.git@main#egg=humanoid_bench"

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/humanoidbench_gl.sh" <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF

cat > "$DEACTIVATE_DIR/humanoidbench_gl.sh" <<'EOF'
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
EOF

log "Setup complete! Activate with: conda activate $ENV_NAME"
