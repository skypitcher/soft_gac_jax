import os
import platform
import socket
import subprocess
import time
import traceback
from pathlib import Path

import jax
import wandb
from stable_baselines3.common.logger import configure as configure_sb3_logger

from common.config import config_to_plain_dict
from common.config_utils import cfg_get
from common.env_utils import make_env, make_eval_env
from common.experiment import (
    make_log_context,
    parse_env_name,
    setup_experiment_dir,
)
from common.logger import setup_logger
from common.run_label import resolved_run_label
from models.utils import is_slurm_job

logger = setup_logger()
RUN_ROOT = Path.cwd().resolve()
WANDB_RUN_FILES = (
    "code_snapshot.zip",
    "config.log",
    "training.log",
    "train.csv",
    "eval.csv",
)


def _flush_logger_handlers() -> None:
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _ensure_wandb_run_files(result_path: str) -> None:
    result_dir = Path(result_path)
    for name in ("train.csv", "eval.csv"):
        path = result_dir / name
        if not path.exists():
            path.touch()


def _save_wandb_run_files(result_path: str, *, policy: str) -> None:
    if wandb.run is None:
        return

    _flush_logger_handlers()
    result_dir = Path(result_path)
    files = [result_dir / name for name in WANDB_RUN_FILES]
    files.extend(sorted(result_dir.glob("*.csv")))
    for path in dict.fromkeys(files):
        if path.is_file():
            wandb.save(str(path), base_path=str(result_dir), policy=policy)


def _space_flatdim(space) -> int:
    from gymnasium.spaces.utils import flatdim

    return int(flatdim(space))


def _fmt_model_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _hostname() -> str:
    host = socket.gethostname() or platform.node() or "unknown"
    return host.split(".", 1)[0]


def _log_system_info(cfg) -> dict[str, str]:
    host = _hostname()
    logger.info("Host: %s", host)
    logger.info("JAX %s │ backend=%s │ devices=%s", jax.__version__, jax.default_backend(), jax.devices())
    logger.info("Python %s │ OS: %s", platform.python_version(), platform.platform())

    gpu_info = "N/A"
    cuda_version = "N/A"
    cpu_name = "N/A"
    try:
        gpu_info = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                timeout=5,
            )
            .decode()
            .strip()
        )
        logger.info("GPU: %s", gpu_info)
    except Exception:
        pass

    try:
        cuda_out = subprocess.check_output(["nvcc", "--version"], timeout=5).decode()
        for line in cuda_out.splitlines():
            if "release" in line:
                cuda_version = line.strip()
                logger.info("CUDA: %s", cuda_version)
                break
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if cpu_name == "N/A":
        cpu_name = platform.processor() or platform.machine() or "N/A"
    logger.info("CPU: %s", cpu_name)

    if cfg.wandb.enabled and wandb.run is not None:
        wandb.config.update(
            {
                "jax_version": jax.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "host": host,
                "cpu": cpu_name,
                "gpu": gpu_info,
                "cuda": cuda_version,
            },
            allow_val_change=True,
        )
        wandb.summary["host"] = host

    return {
        "host": host,
        "gpu": gpu_info,
        "cuda": cuda_version,
        "cpu": cpu_name,
    }


def _log_model_summary(model) -> None:
    obs_dim = _space_flatdim(model.observation_space)
    act_dim = _space_flatdim(model.action_space)

    if hasattr(model, "describe_model"):
        info = model.describe_model()
    else:
        info = {}

    actor_params = int(info.get("actor_params", 0))
    critic_params = int(info.get("critic_params", 0))
    summary_parts = [f"{key}={_fmt_model_value(value)}" for key, value in info.items() if key not in {"actor_params", "critic_params"}]
    logger.info(
        "Model │ obs_dim=%d │ act_dim=%d │ actor: %.5fK │ critic: %.5fK%s",
        obs_dim,
        act_dim,
        actor_params / 1e3,
        critic_params / 1e3,
        f" │ {' │ '.join(summary_parts)}" if summary_parts else "",
    )


def _wandb_project(cfg) -> str:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = parse_env_name(env_name)
    domain = domain.replace("/", "_").replace("-", "_")
    task = task.replace("/", "_").replace("-", "_")
    prefix = str(cfg.wandb.project) if getattr(cfg.wandb, "project", None) else "SoftGAC"
    return f"{prefix}-{domain}-{task}" if domain else f"{prefix}-{task}"


def _wandb_group(cfg) -> str:
    return str(cfg.run_label)


def _wandb_job_type(cfg) -> str:
    return str(cfg.wandb.job_type)


def _wandb_run_name(cfg) -> str:
    return f"{cfg.run_label}_seed{cfg.seed}"


def _create_alg(cfg, result_path: str, eval_complete_cb=None):
    import gymnasium as gym

    from algorithms.soft_gac import SoftGAC
    from algorithms.sac import SAC
    from algorithms.dime import DIME
    from algorithms.qsm import QSM
    from algorithms.flowrl import FlowRL
    from algorithms.flac import FLAC
    from algorithms.qvpo import QVPO
    from common.buffers import DMCCompatibleDictReplayBuffer
    from models.actor_critic_evaluation_callback import EvalCallback

    try:
        import myosuite  # noqa: F401
    except ImportError:
        print("myosuite not installed")

    env_kwargs = config_to_plain_dict(getattr(cfg, "env_kwargs", {})) or {}
    training_env = make_env(cfg.env_name, seed=cfg.seed, env_kwargs=env_kwargs)
    eval_env = make_eval_env(cfg.env_name, seed=cfg.seed, env_kwargs=env_kwargs)
    rb_class = None
    if cfg.env_name.startswith("dm_control/") and isinstance(training_env.observation_space, gym.spaces.Dict):
        rb_class = DMCCompatibleDictReplayBuffer

    alg_name = str(cfg.alg.name)
    sb3_algos = {
        "soft_gac": SoftGAC,
        "sac": SAC,
        "dime": DIME,
        "qsm": QSM,
        "flowrl": FlowRL,
        "flac": FLAC,
        "qvpo": QVPO,
    }
    alg_cls = sb3_algos.get(alg_name)
    if alg_cls is None:
        raise ValueError(f"Unsupported algorithm: {alg_name}. Expected one of {sorted(sb3_algos)}")

    model = alg_cls(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env=training_env,
        model_save_path=os.path.join(result_path, "checkpoint") if alg_name == "soft_gac" else None,
        save_every_n_steps=max(1, int(cfg.tot_time_steps / 100000)),
        cfg=cfg,
        tensorboard_log=None,
        replay_buffer_class=rb_class,
    )

    log_context = make_log_context(cfg.env_name)
    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=cfg.seed,
        best_model_save_path=None,
        log_path=None,
        csv_path=os.path.join(result_path, "eval.csv"),
        eval_freq=max(int(cfg.eval_freq), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        logger_obj=logger,
        log_context=log_context,
        on_eval_complete=eval_complete_cb,
    )
    return model, eval_callback


def initialize_and_run(cfg, *, entry_script: str = "main.py") -> str:
    cfg.run_label = resolved_run_label(cfg)
    result_path, snapshot_path = setup_experiment_dir(cfg, cfg.run_label, entry_script=entry_script)
    training_log_path = os.path.join(result_path, "training.log")
    setup_logger(log_file=training_log_path, capture_stdio=False)

    if cfg.wandb.enabled:
        wandb.init(
            settings=wandb.Settings(_service_wait=300, console="redirect", console_multipart=True),
            project=_wandb_project(cfg),
            group=_wandb_group(cfg),
            job_type=_wandb_job_type(cfg),
            name=_wandb_run_name(cfg),
            config=config_to_plain_dict(cfg),
            entity=cfg.wandb.entity,
            sync_tensorboard=True,
            dir=str(RUN_ROOT),
        )
        wandb.config.update(
            {
                "result_path": result_path,
                "snapshot_path": snapshot_path,
            },
            allow_val_change=True,
        )
        wandb.summary["result_path"] = result_path
    setup_logger(log_file=training_log_path, capture_stdio=True)

    if cfg.wandb.enabled:
        _ensure_wandb_run_files(result_path)
        if is_slurm_job():
            logger.info("SLURM_JOB_ID: %s", os.environ.get("SLURM_JOB_ID"))
            wandb.summary["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")

    logger.info("Run start │ script=%s │ env=%s │ seed=%s │ host=%s", entry_script, cfg.env_name, cfg.seed, _hostname())
    logger.info("Result path │ %s", result_path)
    logger.info("Code snapshot │ %s", snapshot_path)

    cfg.log(logger)
    _log_system_info(cfg)

    from stable_baselines3.common.callbacks import CallbackList
    from wandb.integration.sb3 import WandbCallback

    from models.episode_logging_callback import EpisodeLoggingCallback
    from models.checkpoint_callback import JaxCheckpointCallback

    model, eval_callback = _create_alg(cfg, result_path, eval_complete_cb=None)
    model.set_logger(configure_sb3_logger(result_path, ["tensorboard"]))
    _log_model_summary(model)
    episode_callback = EpisodeLoggingCallback(verbose=0, csv_path=os.path.join(result_path, "train.csv"))
    callbacks = [episode_callback, eval_callback]

    checkpoint_enabled = bool(cfg_get(cfg, "checkpoint.enabled", default=True))
    if str(cfg.alg.name) == "soft_gac" and checkpoint_enabled:
        checkpoint_freq = cfg_get(cfg, "checkpoint.save_freq", default=None)
        checkpoint_keep_last = cfg_get(cfg, "checkpoint.keep_last", default=2)
        checkpoint_save_at_end = bool(cfg_get(cfg, "checkpoint.save_at_end", default=True))
        callbacks.append(
            JaxCheckpointCallback(
                os.path.join(result_path, "checkpoint"),
                save_freq=checkpoint_freq,
                keep_last=checkpoint_keep_last,
                save_at_end=checkpoint_save_at_end,
            )
        )
        logger.info(
            "Checkpoint enabled │ dir=%s │ save_freq=%s │ keep_last=%s │ save_at_end=%s",
            os.path.join(result_path, "checkpoint"),
            checkpoint_freq,
            checkpoint_keep_last,
            checkpoint_save_at_end,
        )

    if cfg.wandb.enabled:
        callback_list = CallbackList([*callbacks, WandbCallback(verbose=0)])
    else:
        callback_list = CallbackList(callbacks)

    model.learn(
        total_timesteps=cfg.tot_time_steps,
        progress_bar=True,
        callback=callback_list,
    )

    return result_path


def run_main(cfg, *, entry_script: str) -> None:
    try:
        starting_time = time.time()
        if cfg.use_jit:
            result_path = initialize_and_run(cfg, entry_script=entry_script)
        else:
            with jax.disable_jit():
                result_path = initialize_and_run(cfg, entry_script=entry_script)
        end_time = time.time()
        logger.info("Training took: %.4f hours", (end_time - starting_time) / 3600)
        if cfg.wandb.enabled:
            _save_wandb_run_files(result_path, policy="now")
            wandb.finish()
    except Exception as ex:
        logger.exception("Unhandled exception")
        traceback.print_tb(ex.__traceback__)
        traceback.print_exception(ex)
        if getattr(cfg, "wandb", None) and cfg.wandb.enabled:
            try:
                if "result_path" in locals():
                    _save_wandb_run_files(result_path, policy="now")
            except Exception:
                logger.exception("Failed to sync W&B run files after exception")
            wandb.finish()
