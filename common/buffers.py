from typing import Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

####### This class overwrites the DictReplayBuffer from stable baselines. It throws an exception when running the DMC's
# humanoid tasks because of the head height observation. Either shimmy or dmc returns it as a 1-dim
# observation, which is not aligned with the general framework. This class only takes care of dimensonality issues of
# observations during sampling from the buffer and doesn't change anything else


class DMCCompatibleDictReplayBuffer(DictReplayBuffer):
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # type: ignore[signature-mismatch]
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: np.atleast_3d(obs)[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        next_obs_ = self._normalize_obs(
            {key: np.atleast_3d(obs)[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


def sample_mixed_recent(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    env: Optional[VecNormalize] = None,
    recent_ratio: float = 0.2,
    recent_window: int = 4096,
):
    """Sample a batch using uniform + recent replay mixture.

    - uniform part samples with replacement from all valid entries
    - recent part samples with replacement from the latest window
    """
    recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
    recent_window = max(1, int(recent_window))
    num_recent = int(round(int(batch_size) * recent_ratio))
    num_recent = min(max(num_recent, 0), int(batch_size))
    num_uniform = int(batch_size) - num_recent

    valid_size = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
    if valid_size <= 0:
        raise ValueError("Cannot sample from an empty replay buffer")

    if num_recent == 0:
        return replay_buffer.sample(batch_size, env=env)

    if num_uniform == 0:
        uniform_idx = np.empty(0, dtype=np.int64)
    else:
        uniform_idx = np.random.randint(0, valid_size, size=num_uniform)

    recent_window = min(recent_window, valid_size)
    if replay_buffer.full:
        recent_start = (replay_buffer.pos - recent_window) % replay_buffer.buffer_size
        if recent_start < replay_buffer.pos:
            recent_pool = np.arange(recent_start, replay_buffer.pos)
        else:
            recent_pool = np.concatenate(
                [
                    np.arange(recent_start, replay_buffer.buffer_size),
                    np.arange(0, replay_buffer.pos),
                ]
            )
    else:
        recent_pool = np.arange(valid_size - recent_window, valid_size)

    recent_idx = recent_pool[np.random.randint(0, len(recent_pool), size=num_recent)]
    batch_inds = np.concatenate([uniform_idx, recent_idx])
    return replay_buffer._get_samples(batch_inds, env=env)


def _concat_sample_fields(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: _concat_sample_fields([value[key] for value in values]) for key in first}
    return th.cat(list(values), dim=0)


def concat_replay_samples(samples):
    if not samples:
        raise ValueError("Cannot concatenate an empty replay sample list")
    sample_type = type(samples[0])
    return sample_type(
        observations=_concat_sample_fields([sample.observations for sample in samples]),
        actions=_concat_sample_fields([sample.actions for sample in samples]),
        next_observations=_concat_sample_fields([sample.next_observations for sample in samples]),
        dones=_concat_sample_fields([sample.dones for sample in samples]),
        rewards=_concat_sample_fields([sample.rewards for sample in samples]),
    )


def sample_replay_per_step(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gradient_steps: int,
    env: Optional[VecNormalize] = None,
    mix_recent: bool = False,
    recent_ratio: float = 0.2,
    recent_window: int = 4096,
):
    """Sample one replay batch per gradient step, then concatenate.

    Fused JIT train loops later split the concatenated batch into
    ``gradient_steps`` chunks. Sampling each chunk independently preserves the
    intended recent replay ratio for every update.
    """
    batch_size = int(batch_size)
    gradient_steps = int(gradient_steps)
    if gradient_steps <= 0:
        raise ValueError("gradient_steps must be positive")

    samples = []
    for _ in range(gradient_steps):
        if mix_recent:
            sample = sample_mixed_recent(
                replay_buffer,
                batch_size,
                env=env,
                recent_ratio=recent_ratio,
                recent_window=recent_window,
            )
        else:
            sample = replay_buffer.sample(batch_size, env=env)
        samples.append(sample)
    return concat_replay_samples(samples)
