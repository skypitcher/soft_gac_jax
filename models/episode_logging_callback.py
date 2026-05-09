import csv
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from common.logger import LOGGER_NAME, log_episode, log_train_metrics
import logging


class EpisodeLoggingCallback(BaseCallback):
    _TRAIN_METRIC_PRIORITY = (
        "n_updates",
        "critic_loss",
        "actor_loss",
        "buffer_critic_loss",
        "value_buffer_loss",
        "q_buffer_mean",
        "q_buffer_std",
        "q_mean",
        "q_std",
        "current_q_values",
        "next_q_values",
        "alpha",
        "alpha_loss",
        "actor_control_cost",
        "next_control_cost",
        "discover_step_norm",
        "soft_q_mean",
        "soft_q_improved_mean",
        "q_improve",
        "teacher_ess",
        "teacher_w_max",
        "teacher_w_min",
    )
    _TRAIN_METRIC_RANK = {name: idx for idx, name in enumerate(_TRAIN_METRIC_PRIORITY)}

    def __init__(self, verbose: int = 0, csv_path: str | None = None):
        super().__init__(verbose=verbose)
        self.logger_obj = logging.getLogger(LOGGER_NAME)
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._episode_rewards = None
        self._episode_steps = None
        self._episode_start_times = None
        self._training_start_time = None
        self._episode_count = 0

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        now = time.time()
        self._episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self._episode_steps = np.zeros(n_envs, dtype=np.int64)
        self._episode_start_times = np.full(n_envs, now, dtype=np.float64)
        self._training_start_time = now
        if self.csv_path is not None:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            file_exists = os.path.isfile(self.csv_path)
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf8")
            self._csv_writer = csv.writer(self._csv_file)
            if not file_exists or os.path.getsize(self.csv_path) == 0:
                self._csv_writer.writerow(["episode", "total_steps", "episode_steps", "reward", "elapsed", "wallclock"])
                self._csv_file.flush()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)

        if rewards.size == 0 or dones.size == 0:
            return True

        self._episode_rewards[: rewards.size] += rewards
        self._episode_steps[: rewards.size] += 1

        now = time.time()
        for env_idx, done in enumerate(dones):
            if not done:
                continue

            self._episode_count += 1
            elapsed = now - float(self._episode_start_times[env_idx])
            total_elapsed = now - float(self._training_start_time)
            log_episode(
                self.logger_obj,
                self._episode_count,
                self.num_timesteps,
                int(self._episode_steps[env_idx]),
                float(self._episode_rewards[env_idx]),
                elapsed,
                total_elapsed,
            )
            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    [
                        self._episode_count,
                        self.num_timesteps,
                        int(self._episode_steps[env_idx]),
                        float(self._episode_rewards[env_idx]),
                        f"{elapsed:.2f}",
                        f"{total_elapsed:.2f}",
                    ]
                )
                self._csv_file.flush()
            self._episode_rewards[env_idx] = 0.0
            self._episode_steps[env_idx] = 0
            self._episode_start_times[env_idx] = now
            self._log_train_snapshot()

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _log_train_snapshot(self) -> None:
        sb3_logger = getattr(self.model, "logger", None)
        if sb3_logger is None:
            return

        name_to_value = getattr(sb3_logger, "name_to_value", None)
        if not name_to_value:
            return

        def metric_suffix(key: str) -> str:
            name = key.removeprefix("train/")
            return name.rsplit("/", 1)[-1]

        def sort_key(key: str):
            suffix = metric_suffix(key)
            return (self._TRAIN_METRIC_RANK.get(suffix, len(self._TRAIN_METRIC_RANK)), key)

        metrics = {}
        for key in sorted((key for key in name_to_value if key.startswith("train/")), key=sort_key):
            value = name_to_value[key]
            if isinstance(value, np.ndarray):
                if value.ndim != 0:
                    continue
                value = value.item()
            metrics[key.removeprefix("train/")] = value

        log_train_metrics(self.logger_obj, metrics)
