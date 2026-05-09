#!/bin/bash
# Local multi-seed launcher for SoftGAC experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SEEDS="0,1,2,3,4,5,6,7"
RUN_SEEDS_CONDA_ENV="${RUN_SEEDS_CONDA_ENV:-${CONDA_DEFAULT_ENV:-soft_gac}}"

usage() {
    cat <<'EOF'
Usage:
  bash run_seeds_new.sh [--script main.py] [--seeds 0,1,2,3] [-- config overrides ...]

Examples:
  bash run_seeds_new.sh alg=soft_gac env_name=dm_control/dog-run
  bash run_seeds_new.sh --seeds 0,1 -- alg=sac env_name=Pendulum-v1

Do not pass seed=... manually; this launcher appends one seed per subprocess.
EOF
}

parse_seeds() {
    local raw="$1"
    local item
    IFS=',' read -r -a SEEDS <<< "$raw"
    [[ ${#SEEDS[@]} -gt 0 ]] || { echo "Empty --seeds list." >&2; exit 1; }
    for item in "${SEEDS[@]}"; do
        [[ "$item" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $item" >&2; exit 1; }
    done
}

maybe_activate_conda_env() {
    if [[ -z "${CONDA_PREFIX:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$RUN_SEEDS_CONDA_ENV"
    fi
}

run_script="main.py"
seed_arg="$DEFAULT_SEEDS"
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            RUN_ARGS+=("$@")
            break
            ;;
        --script)
            run_script="$2"
            shift 2
            ;;
        --script=*)
            run_script="${1#*=}"
            shift
            ;;
        --seeds)
            seed_arg="$2"
            shift 2
            ;;
        --seeds=*)
            seed_arg="${1#*=}"
            shift
            ;;
        seed=*|+seed=*|++seed=*)
            echo "Do not pass seed=... manually; use --seeds." >&2
            exit 1
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

parse_seeds "$seed_arg"
maybe_activate_conda_env

if [[ "$run_script" != /* ]]; then
    RUN_SCRIPT_PATH="$SCRIPT_DIR/$run_script"
else
    RUN_SCRIPT_PATH="$run_script"
fi
[[ -f "$RUN_SCRIPT_PATH" ]] || { echo "Run script not found: $RUN_SCRIPT_PATH" >&2; exit 1; }

mkdir -p "$SCRIPT_DIR/logs"

echo "Launching ${#SEEDS[@]} seeds"
echo "Script: $RUN_SCRIPT_PATH"
echo "Seeds: ${SEEDS[*]}"
echo "Args: ${RUN_ARGS[*]}"
echo "Monitor: tail -f $SCRIPT_DIR/logs/main_*_seed*.log"

PIDS=()
FAILED=0
LAUNCH_ID="$(date +%Y%m%d_%H%M%S)"

for seed in "${SEEDS[@]}"; do
    log_file="$SCRIPT_DIR/logs/main_${LAUNCH_ID}_seed${seed}.log"
    echo "  [seed=$seed] -> $log_file"
    (
        cd "$SCRIPT_DIR"
        OMP_NUM_THREADS=1 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python "$RUN_SCRIPT_PATH" "${RUN_ARGS[@]}" "seed=$seed"
    ) > "$log_file" 2>&1 &
    PIDS+=("$!")
done

for i in "${!PIDS[@]}"; do
    seed="${SEEDS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  [seed=$seed] done"
    else
        status=$?
        echo "  [seed=$seed] FAILED (exit $status)"
        FAILED=$((FAILED + 1))
    fi
done

if (( FAILED > 0 )); then
    echo "$FAILED / ${#SEEDS[@]} seeds failed."
    exit 1
fi

echo "All seeds completed."
