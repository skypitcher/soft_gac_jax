import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
gpu_det_flag = "--xla_gpu_deterministic_ops=true"
current_xla_flags = os.environ.get("XLA_FLAGS", "")
if gpu_det_flag not in current_xla_flags:
    os.environ["XLA_FLAGS"] = f"{current_xla_flags} {gpu_det_flag}".strip()

from common.config import load_config


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv)
    from runner import run_main

    run_main(cfg, entry_script="main.py")


if __name__ == "__main__":
    main()
