import logging
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from common.logger import LOGGER_NAME


class JaxCheckpointCallback(BaseCallback):
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        save_freq: int | None = None,
        keep_last: int | None = None,
        save_at_end: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = int(save_freq or 0)
        self.keep_last = keep_last
        self.save_at_end = bool(save_at_end)
        self._last_saved_step = -1
        self.logger_obj = logging.getLogger(LOGGER_NAME)

    def _save(self, reason: str) -> None:
        save_checkpoint = getattr(self.model, "save_checkpoint", None)
        if save_checkpoint is None:
            return
        if self.num_timesteps == self._last_saved_step and reason != "final":
            return
        path = save_checkpoint(self.checkpoint_dir, keep_last=self.keep_last, reason=reason)
        self._last_saved_step = self.num_timesteps
        self.logger_obj.info("Checkpoint saved │ step=%s │ %s", self.num_timesteps, path)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save("periodic")
        return True

    def _on_training_end(self) -> None:
        if self.save_at_end:
            self._save("final")
