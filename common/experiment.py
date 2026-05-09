import datetime
import zipfile
from pathlib import Path
from types import SimpleNamespace

from common.config import config_to_text

RUN_ROOT = Path.cwd().resolve()


def parse_env_name(env_name: str) -> tuple[str, str]:
    env_name = str(env_name)
    if env_name.startswith("dm_control/"):
        domain_task = env_name.split("/", 1)[1]
        if "-" in domain_task:
            domain, task = domain_task.split("-", 1)
        else:
            domain, task = "dm_control", domain_task
        return domain, task
    if "/" in env_name:
        return tuple(env_name.split("/", 1))
    return "", env_name


def make_log_context(env_name: str):
    domain, task = parse_env_name(env_name)
    return SimpleNamespace(domain=domain, task=task)


def _write_code_snapshot(snapshot_path: Path, entry_script: str | None = None) -> None:
    excluded_dirs = {"__pycache__", "results", "logs", "wandb", ".git", ".venv", "venv"}
    included_suffixes = {".py", ".yaml", ".sh", ".md", ".txt"}
    included_names = {"LICENSE", ".gitignore"}

    with zipfile.ZipFile(snapshot_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in RUN_ROOT.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in excluded_dirs for part in file_path.parts):
                continue
            if file_path.suffix not in included_suffixes and file_path.name not in included_names:
                continue
            rel = file_path.relative_to(RUN_ROOT)
            zf.write(file_path, rel)


def setup_experiment_dir(cfg, label: str, entry_script: str | None = None) -> tuple[str, str]:
    domain, task = parse_env_name(cfg.env_name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_root_value = getattr(cfg, "result_root", None)
    result_root = Path(result_root_value) if result_root_value else Path("results")
    env_dir = f"{domain}-{task}" if domain else task
    result_path = result_root / env_dir / label / f"{timestamp}_seed{cfg.seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    config_log = result_path / "config.log"
    config_log.write_text(config_to_text(cfg), encoding="utf-8")

    snapshot_path = result_path / "code_snapshot.zip"
    _write_code_snapshot(snapshot_path, entry_script=entry_script)
    return str(result_path), str(snapshot_path)
