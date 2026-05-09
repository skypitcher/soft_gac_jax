"""QVPO: Q-weighted Variational Policy Optimization (arXiv:2405.16173).

paper: https://arxiv.org/pdf/2405.16173
code: https://github.com/wadx2019/qvpo/
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv

from common.buffers import sample_mixed_recent, sample_replay_per_step
from common.crossq_c51 import twin_q_expectation, update_crossq_c51_critic
from common.label_utils import replay_mix_recent_label
from common.off_policy_algorithm import OffPolicyAlgorithmJax
from models.legacy_utils import soft_update
from models.qvpo_policy import QVPOPolicy


def _sanitize_label_part(value: str) -> str:
    return str(value).replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


class QVPO(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[QVPOPolicy]]] = {
        "MlpPolicy": QVPOPolicy,
        "MultiInputPolicy": QVPOPolicy,
    }
    policy: QVPOPolicy
    action_space: spaces.Box

    @staticmethod
    def default_run_label(cfg) -> str:
        domain, task = _env_parts(cfg)
        env_label = f"{domain}_{task}" if domain else task
        critic_hs = [int(h) for h in cfg.alg.critic.hs]
        critic_layers = len(critic_hs)
        critic_hidden = str(critic_hs[0]) if len(set(critic_hs)) == 1 else "-".join(str(h) for h in critic_hs)
        critic_label = f"cxq1_c{critic_hidden}x{critic_layers}"
        weighted = "w" if bool(cfg.alg.qvpo.weighted) else "uw"
        aug = "aug" if bool(cfg.alg.qvpo.aug) else "mem"
        return (
            f"qvpo_{env_label}_"
            f"nfe{int(cfg.alg.actor.n_timesteps)}_"
            f"a{int(cfg.alg.actor.hidden_size)}x{int(cfg.alg.actor.num_layers)}_"
            f"{weighted}_{aug}_"
            f"pd{int(cfg.alg.train.policy_delay)}_"
            f"{critic_label}_"
            f"{replay_mix_recent_label(cfg)}"
        )

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        model_save_path: str | None,
        save_every_n_steps: int,
        cfg,
        train_freq: Union[int, Tuple[int, str]] = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        stats_window_size: int = 100,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=float(cfg.alg.optimizer.lr_actor),
            qf_learning_rate=float(cfg.alg.optimizer.lr_critic),
            buffer_size=int(cfg.alg.replay.buffer_size),
            learning_starts=int(cfg.alg.train.learning_starts),
            batch_size=int(cfg.alg.train.batch_size),
            tau=1.0,
            gamma=float(cfg.alg.train.gamma),
            train_freq=train_freq,
            gradient_steps=int(cfg.alg.train.utd),
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=int(cfg.seed),
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )
        self.cfg = cfg
        self.model_save_path = model_save_path
        self.save_every_n_steps = save_every_n_steps
        self.mix_recent_replay = bool(cfg.alg.replay.mix_recent)
        self.recent_replay_ratio = float(cfg.alg.replay.recent_ratio)
        self.recent_replay_window = int(cfg.alg.replay.recent_window)
        self.policy_delay = int(cfg.alg.train.policy_delay)
        self.target_update_interval = int(cfg.alg.target.target_update_interval)
        self.actor_tau = float(cfg.alg.target.actor_tau)

        self._T = int(cfg.alg.actor.n_timesteps)
        self._d_a = int(np.prod(self.action_space.shape))
        self.action_high = jnp.ones((self._d_a,), dtype=jnp.float32)
        self.action_low = -jnp.ones((self._d_a,), dtype=jnp.float32)
        self._noise_ratio = float(cfg.alg.qvpo.noise_ratio)
        self._clip_denoised = bool(cfg.alg.qvpo.clip_denoised)
        self._weighted = bool(cfg.alg.qvpo.weighted)
        self._aug = bool(cfg.alg.qvpo.aug)
        self._gradient = bool(cfg.alg.qvpo.gradient)
        self._train_sample = int(cfg.alg.qvpo.train_sample)
        self._chosen = int(cfg.alg.qvpo.chosen)
        self._behavior_sample = int(cfg.alg.qvpo.behavior_sample)
        self._target_sample = int(cfg.alg.qvpo.target_sample)
        self._q_transform = str(cfg.alg.qvpo.q_transform).lower()
        self._q_neg = float(cfg.alg.qvpo.q_neg)
        self._cut = float(cfg.alg.qvpo.cut)
        self._beta = float(cfg.alg.qvpo.beta)
        self._alpha_mean = float(cfg.alg.qvpo.alpha_mean)
        self._alpha_std = float(cfg.alg.qvpo.alpha_std)
        self._entropy_alpha = float(cfg.alg.qvpo.entropy_alpha)
        self._entropy_repeats = int(cfg.alg.qvpo.entropy_repeats)
        self._action_lr = float(cfg.alg.qvpo.action_lr)
        self._action_gradient_steps = int(cfg.alg.qvpo.action_gradient_steps)
        self._action_grad_norm = self._d_a * float(cfg.alg.qvpo.action_grad_norm_ratio)
        self.action_optimizer = optax.adam(self._action_lr, eps=1e-5)

        self._diffusion_obs: np.ndarray | None = None
        self._diffusion_actions: np.ndarray | None = None
        self._diffusion_pos = 0
        self._diffusion_full = False

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self, reset: bool = False) -> None:
        if not reset:
            super()._setup_model()
        if not hasattr(self, "policy") or self.policy is None or reset:
            self.policy = self.policy_class(self.observation_space, self.action_space, self.cfg)
            assert isinstance(self.qf_learning_rate, float)
            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)
            self.critic_module = self.policy.critic_module
            self.actor_module = self.policy.actor_module
            self.actor_optimizer = self.policy.actor_optimizer
            self._sqrt_alphas_cumprod = self.policy._sqrt_alphas_cumprod
            self._sqrt_one_minus_alphas_cumprod = self.policy._sqrt_one_minus_alphas_cumprod
            self._sqrt_recip_alphas_cumprod = self.policy._sqrt_recip_alphas_cumprod
            self._sqrt_recipm1_alphas_cumprod = self.policy._sqrt_recipm1_alphas_cumprod
            self._posterior_mean_coef1 = self.policy._posterior_mean_coef1
            self._posterior_mean_coef2 = self.policy._posterior_mean_coef2
            self._posterior_log_variance_clipped = self.policy._posterior_log_variance_clipped
            self._num_atoms = int(self.cfg.alg.critic.n_atoms)
            self._v_min = float(self.cfg.alg.critic.v_min)
            self._v_max = float(self.cfg.alg.critic.v_max)
            self._z_atoms = jnp.linspace(self._v_min, self._v_max, self._num_atoms)
            self._init_diffusion_memory()

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in self.cfg.alg.critic.hs]
        return {
            "family": "qvpo",
            "actor_hidden_size": int(self.cfg.alg.actor.hidden_size),
            "actor_num_layers": int(self.cfg.alg.actor.num_layers),
            "diffusion_timesteps": int(self._T),
            "time_embed_dim": int(self.cfg.alg.actor.time_embed_dim),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "critic_crossq": True,
            "critic_update": "crossq_c51",
            "policy_delay": int(self.policy_delay),
            "target_update_interval": int(self.target_update_interval),
            "actor_tau": float(self.actor_tau),
            "weighted": bool(self._weighted),
            "aug": bool(self._aug),
            "gradient": bool(self._gradient),
            "train_sample": int(self._train_sample),
            "chosen": int(self._chosen),
            "behavior_sample": int(self._behavior_sample),
            "target_sample": int(self._target_sample),
            "eval_sample": int(self.cfg.alg.qvpo.eval_sample),
            "q_transform": str(self._q_transform),
            "entropy_alpha": float(self._entropy_alpha),
            "action_lr": float(self._action_lr),
            "action_gradient_steps": int(self._action_gradient_steps),
            "action_grad_norm": float(self._action_grad_norm),
            "noise_ratio": float(self._noise_ratio),
            "beta_schedule": str(self.cfg.alg.qvpo.beta_schedule),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "actor_params": self._count_params(self.policy.state.actor_params),
            "critic_params": self._count_params(self.policy.state.qf_state.params),
        }

    def _init_diffusion_memory(self) -> None:
        if isinstance(self.observation_space, spaces.Dict):
            obs_dim = sum(int(np.prod(space.shape)) for space in self.observation_space.spaces.values())
        else:
            obs_dim = int(np.prod(self.observation_space.shape))
        self._diffusion_obs = np.zeros((self.buffer_size, obs_dim), dtype=np.float32)
        self._diffusion_actions = np.zeros((self.buffer_size, self._d_a), dtype=np.float32)
        self._diffusion_pos = 0
        self._diffusion_full = False

    def _flatten_obs_batch(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            parts = []
            for key, space in self.observation_space.spaces.items():
                arr = np.asarray(obs[key])
                if arr.ndim == len(space.shape):
                    arr = arr.reshape((1, *space.shape))
                parts.append(arr.reshape((arr.shape[0], -1)))
            return np.concatenate(parts, axis=1).astype(np.float32)
        arr = np.asarray(obs)
        if arr.ndim == len(self.observation_space.shape):
            arr = arr.reshape((1, *self.observation_space.shape))
        return arr.reshape((arr.shape[0], -1)).astype(np.float32)

    def _append_diffusion_memory(self, obs, action) -> None:
        assert self._diffusion_obs is not None and self._diffusion_actions is not None
        obs_batch = self._flatten_obs_batch(obs)
        action_batch = np.asarray(action, dtype=np.float32).reshape((obs_batch.shape[0], -1))
        for i in range(obs_batch.shape[0]):
            self._diffusion_obs[self._diffusion_pos] = obs_batch[i]
            self._diffusion_actions[self._diffusion_pos] = action_batch[i]
            self._diffusion_pos = (self._diffusion_pos + 1) % self.buffer_size
            if self._diffusion_pos == 0:
                self._diffusion_full = True

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        last_obs = getattr(self, "_last_original_obs", None)
        if last_obs is None:
            last_obs = self._last_obs
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        self._append_diffusion_memory(last_obs, buffer_action)

    def _sample_diffusion_memory(self, batch_size: int):
        assert self._diffusion_obs is not None and self._diffusion_actions is not None
        valid_size = self.buffer_size if self._diffusion_full else self._diffusion_pos
        if valid_size <= 0:
            raise ValueError("Cannot sample from an empty QVPO diffusion memory")
        idx = np.random.randint(0, valid_size, size=int(batch_size))
        return idx, jnp.array(self._diffusion_obs[idx]), jnp.array(self._diffusion_actions[idx])

    def _replace_diffusion_actions(self, idx, actions) -> None:
        assert self._diffusion_actions is not None
        self._diffusion_actions[idx] = np.asarray(actions, dtype=np.float32)

    def _current_entropy_alpha(self) -> float:
        """Official QVPO linearly decays entropy augmentation to a small floor."""
        base = float(self._entropy_alpha)
        if base <= 0.0:
            return 0.0
        total_steps = max(float(getattr(self.cfg, "tot_time_steps", getattr(self.cfg, "iters", 1))), 1.0)
        progress = min(max(float(self.num_timesteps) / total_steps, 0.0), 1.0)
        return min(base, max(0.002, base * (1.0 - progress)))

    def _prepare_batch(self, data):
        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()
        return (
            jnp.array(obs, dtype=jnp.float32),
            jnp.array(data.actions.numpy(), dtype=jnp.float32),
            jnp.array(data.rewards.numpy(), dtype=jnp.float32).reshape(-1, 1),
            jnp.array(next_obs, dtype=jnp.float32),
            jnp.array(data.dones.numpy(), dtype=jnp.float32).reshape(-1, 1),
        )

    def _sample_replay(self, batch_size: int):
        if self.mix_recent_replay:
            return sample_mixed_recent(
                self.replay_buffer,
                batch_size,
                env=self._vec_normalize_env,
                recent_ratio=self.recent_replay_ratio,
                recent_window=self.recent_replay_window,
            )
        return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

    def _critic_eval(self, qf_state, obs, act):
        return qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            obs,
            act,
            train=False,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reverse_sample(self, actor_params, obs, rng, noise_ratio):
        b = obs.shape[0]
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (b, self._d_a))

        def reverse_step(carry, t_idx):
            x_curr, rng_carry = carry
            t_batch = jnp.full((b,), t_idx, dtype=jnp.int32)
            eps_pred = self.actor_module.apply({"params": actor_params}, x_curr, t_batch, obs)
            x_recon = self._sqrt_recip_alphas_cumprod[t_idx] * x_curr - self._sqrt_recipm1_alphas_cumprod[t_idx] * eps_pred
            if self._clip_denoised:
                x_recon = jnp.clip(x_recon, self.action_low, self.action_high)
            mean = self._posterior_mean_coef1[t_idx] * x_recon + self._posterior_mean_coef2[t_idx] * x_curr
            rng_carry, z_key = jax.random.split(rng_carry)
            z = jax.random.normal(z_key, x_curr.shape)
            nonzero = (t_idx > 0).astype(jnp.float32)
            std = jnp.exp(0.5 * self._posterior_log_variance_clipped[t_idx])
            x_next = mean + nonzero * std * z * noise_ratio
            return (jnp.clip(x_next, self.action_low, self.action_high), rng_carry), None

        t_steps = jnp.arange(self._T - 1, -1, -1)
        (x_final, _), _ = jax.lax.scan(reverse_step, (x, rng), t_steps)
        return jnp.clip(x_final, self.action_low, self.action_high)

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def _sample_best_actions(self, actor_params, qf_state, obs, rng, num_samples, noise_ratio):
        b = obs.shape[0]
        tiled_obs = jnp.repeat(obs[None, :, :], num_samples, axis=0).reshape(num_samples * b, obs.shape[-1])
        actions = self._reverse_sample(actor_params, tiled_obs, rng, noise_ratio)
        q_dist = self._critic_eval(qf_state, tiled_obs, actions)
        _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
        actions = actions.reshape(num_samples, b, self._d_a).transpose(1, 0, 2)
        q_values = min_q.reshape(num_samples, b).T
        best_idx = jnp.argmax(q_values, axis=1)
        return jnp.take_along_axis(actions, best_idx[:, None, None], axis=1)[:, 0, :]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_critic(self, state, batch, rng):
        s, a, r, ns, dones = batch
        next_actions = self._sample_best_actions(
            state.target_actor_params,
            state.qf_state,
            ns,
            rng,
            self._target_sample,
            self._noise_ratio,
        )
        zero_penalty = jnp.zeros((s.shape[0],), dtype=jnp.float32)
        qf_state, metrics, rng, current_q = update_crossq_c51_critic(
            self.gamma,
            state.qf_state,
            s,
            a,
            ns,
            next_actions,
            r,
            dones,
            zero_penalty,
            self._num_atoms,
            self._z_atoms,
            self._v_min,
            self._v_max,
            float(self.cfg.alg.critic.dist_entropy_coeff),
            rng,
        )
        metrics = dict(metrics, target_q_values=metrics["next_q_values"])
        return state.replace(qf_state=qf_state, critic_updates=state.critic_updates + 1), dict(metrics, q_std=current_q.std())

    def _forward_diffusion(self, a_0, t, noise):
        sqrt_alpha_bar = self._sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus = self._sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha_bar * a_0 + sqrt_one_minus * noise

    def _q_transform_weights(self, q, v, running_q_mean, running_q_std, batch_size):
        if self._q_transform == "qrelu":
            return jnp.maximum(q, -self._q_neg) + self._q_neg
        if self._q_transform == "qexpn":
            normalized = jnp.clip((q - running_q_mean) / (running_q_std + 1e-6), -8.0, 8.0)
            return jnp.exp(self._beta * normalized)
        if self._q_transform == "qcut":
            return jnp.maximum(q, self._cut)
        if self._q_transform == "qcut0n":
            return jnp.maximum(self._beta * (q - running_q_mean), 0.0)
        if self._q_transform == "qcut1n":
            return jnp.maximum(self._beta * (q - running_q_mean), 1.0)
        if self._q_transform == "qadv":
            adv = q.reshape(batch_size, self._chosen, 1) - v[:, None, :]
            return jnp.maximum(adv, 0.0).reshape(batch_size * self._chosen, 1)
        raise ValueError(f"Unsupported q_transform='{self._q_transform}'")

    def _clip_action_grads(self, grads):
        if self._action_grad_norm <= 0:
            return grads, jnp.array(0.0)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, self._action_grad_norm / (grad_norm + 1e-6))
        return jax.tree.map(lambda g: g * scale, grads), grad_norm

    @functools.partial(jax.jit, static_argnums=(0,))
    def _action_gradient(self, qf_state, states, actions):
        opt_state = self.action_optimizer.init(actions)

        def step(carry, _):
            curr_actions, curr_opt = carry

            def loss_fn(action_var):
                q_dist = self._critic_eval(qf_state, states, action_var)
                _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
                return -jnp.mean(min_q)

            loss, grads = jax.value_and_grad(loss_fn)(curr_actions)
            grads, grad_norm = self._clip_action_grads(grads)
            updates, next_opt = self.action_optimizer.update(grads, curr_opt, curr_actions)
            next_actions = optax.apply_updates(curr_actions, updates)
            return (jnp.clip(next_actions, self.action_low, self.action_high), next_opt), (loss, grad_norm)

        (best_actions, _), (losses, grad_norms) = jax.lax.scan(
            step,
            (actions, opt_state),
            jnp.arange(self._action_gradient_steps),
        )
        return jax.lax.stop_gradient(best_actions), losses[-1], grad_norms[-1]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_augmented_actions(self, state, states, origin_actions, rng):
        del origin_actions
        b = states.shape[0]
        tiled_states = jnp.repeat(states[None, :, :], self._train_sample, axis=0).reshape(self._train_sample * b, states.shape[-1])
        actions = self._reverse_sample(state.actor_params, tiled_states, rng, self._noise_ratio)
        q_dist = self._critic_eval(state.qf_state, tiled_states, actions)
        _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
        actions_by_state = actions.reshape(self._train_sample, b, self._d_a).transpose(1, 0, 2)
        q_by_state = min_q.reshape(self._train_sample, b, 1).transpose(1, 0, 2)
        q_mean = q_by_state.mean()
        q_std = q_by_state.std()
        v = q_by_state.mean(axis=1)
        top_q, top_idx = jax.lax.top_k(q_by_state[..., 0], self._chosen)
        selected_actions = jnp.take_along_axis(actions_by_state, top_idx[:, :, None], axis=1)
        selected_states = jnp.repeat(states[:, None, :], self._chosen, axis=1)
        return (
            selected_states.reshape(b * self._chosen, states.shape[-1]),
            selected_actions.reshape(b * self._chosen, self._d_a),
            top_q.reshape(b * self._chosen, 1),
            v,
            q_mean,
            q_std,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_actor_aug(self, state, states, origin_actions, rng, entropy_alpha):
        k_sample, k_grad, k_t, k_eps, k_entropy = jax.random.split(rng, 5)
        batch_size = states.shape[0]
        actor_states, actor_actions, q, v, q_mean, q_std = self._sample_augmented_actions(
            state,
            states,
            origin_actions,
            k_sample,
        )
        action_grad_loss = jnp.array(0.0)
        action_grad_norm = jnp.array(0.0)
        if self._gradient:
            actor_actions, action_grad_loss, action_grad_norm = self._action_gradient(state.qf_state, actor_states, actor_actions)
            q_dist = self._critic_eval(state.qf_state, actor_states, actor_actions)
            _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
            q = min_q.reshape(-1, 1)

        new_running_q_std = state.running_q_std + self._alpha_std * (q_std - state.running_q_std)
        new_running_q_mean = state.running_q_mean + self._alpha_mean * (q_mean - state.running_q_mean)
        weights = jnp.ones_like(q)
        if self._weighted:
            weights = self._q_transform_weights(q, v, new_running_q_mean, new_running_q_std, batch_size)

        if self._entropy_alpha > 0.0 and self._entropy_repeats > 0:
            rand_states = jnp.repeat(actor_states[None, :, :], self._entropy_repeats, axis=0).reshape(
                self._entropy_repeats * actor_states.shape[0],
                actor_states.shape[-1],
            )
            rand_actions = jax.random.uniform(
                k_entropy,
                (self._entropy_repeats * actor_actions.shape[0], self._d_a),
                minval=-1.0,
                maxval=1.0,
            )
            rand_weights = jnp.repeat(weights[None, :, :], self._entropy_repeats, axis=0).reshape(-1, 1) * entropy_alpha
            actor_states = jnp.concatenate([actor_states, rand_states], axis=0)
            actor_actions = jnp.concatenate([actor_actions, rand_actions], axis=0)
            weights = jnp.concatenate([weights, rand_weights], axis=0)

        t = jax.random.randint(k_t, (actor_states.shape[0],), 0, self._T)
        eps = jax.random.normal(k_eps, actor_actions.shape)
        noisy_actions = self._forward_diffusion(actor_actions, t, eps)

        def actor_loss_fn(actor_params):
            pred = self.actor_module.apply({"params": actor_params}, noisy_actions, t, actor_states)
            return jnp.mean(weights * ((pred - eps) ** 2))

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(state.actor_params)
        updates, new_opt = self.actor_optimizer.update(grads, state.actor_opt_state, state.actor_params)
        new_actor_params = optax.apply_updates(state.actor_params, updates)
        return state.replace(
            actor_params=new_actor_params,
            actor_opt_state=new_opt,
            running_q_mean=new_running_q_mean,
            running_q_std=new_running_q_std,
            actor_updates=state.actor_updates + 1,
        ), {
            "actor_loss": actor_loss,
            "q_weight_mean": weights.mean(),
            "q_selected_mean": q.mean(),
            "q_selected_std": q.std(),
            "running_q_mean": new_running_q_mean,
            "running_q_std": new_running_q_std,
            "action_gradient_loss": action_grad_loss,
            "action_gradient_norm": action_grad_norm,
            "entropy_alpha": entropy_alpha,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_actor_memory(self, state, states, actions, rng):
        k_grad, k_t, k_eps = jax.random.split(rng, 3)
        best_actions, action_loss, action_grad_norm = self._action_gradient(state.qf_state, states, actions)
        q_dist = self._critic_eval(state.qf_state, states, best_actions)
        _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
        q = min_q.reshape(-1, 1)
        q_mean = q.mean()
        q_std = q.std()
        new_running_q_std = state.running_q_std + self._alpha_std * (q_std - state.running_q_std)
        new_running_q_mean = state.running_q_mean + self._alpha_mean * (q_mean - state.running_q_mean)
        weights = jnp.ones_like(q)
        if self._weighted:
            weights = self._q_transform_weights(q, q.reshape(-1, 1), new_running_q_mean, new_running_q_std, states.shape[0])

        t = jax.random.randint(k_t, (states.shape[0],), 0, self._T)
        eps = jax.random.normal(k_eps, best_actions.shape)
        noisy_actions = self._forward_diffusion(best_actions, t, eps)

        def actor_loss_fn(actor_params):
            pred = self.actor_module.apply({"params": actor_params}, noisy_actions, t, states)
            return jnp.mean(weights * ((pred - eps) ** 2))

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(state.actor_params)
        updates, new_opt = self.actor_optimizer.update(grads, state.actor_opt_state, state.actor_params)
        new_actor_params = optax.apply_updates(state.actor_params, updates)
        return state.replace(
            actor_params=new_actor_params,
            actor_opt_state=new_opt,
            running_q_mean=new_running_q_mean,
            running_q_std=new_running_q_std,
            actor_updates=state.actor_updates + 1,
        ), best_actions, {
            "actor_loss": actor_loss,
            "q_weight_mean": weights.mean(),
            "q_selected_mean": q.mean(),
            "q_selected_std": q.std(),
            "running_q_mean": new_running_q_mean,
            "running_q_std": new_running_q_std,
            "action_gradient_loss": action_loss,
            "action_gradient_norm": action_grad_norm,
            "entropy_alpha": jnp.asarray(0.0),
        }

    @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def _train_fused(
        self,
        state,
        batch,
        gradient_steps: int,
        policy_update_indices: tuple[int, ...],
        target_update_indices: tuple[int, ...],
        rng,
        entropy_alpha,
        memory_states,
        memory_actions,
    ):
        metric_sums = {
            "critic_loss": jnp.asarray(0.0),
            "current_q_values": jnp.asarray(0.0),
            "next_q_values": jnp.asarray(0.0),
            "target_q_values": jnp.asarray(0.0),
            "q_std": jnp.asarray(0.0),
            "actor_loss": jnp.asarray(0.0),
            "q_weight_mean": jnp.asarray(0.0),
            "q_selected_mean": jnp.asarray(0.0),
            "q_selected_std": jnp.asarray(0.0),
            "running_q_mean": jnp.asarray(0.0),
            "running_q_std": jnp.asarray(0.0),
            "action_gradient_loss": jnp.asarray(0.0),
            "action_gradient_norm": jnp.asarray(0.0),
            "entropy_alpha": jnp.asarray(0.0),
        }
        actor_count = jnp.asarray(0.0)
        actor_slot = 0
        updated_memory_actions = []

        for i in range(gradient_steps):
            def slice_batch(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                step_batch_size = x.shape[0] // gradient_steps
                return x[step_batch_size * step: step_batch_size * (step + 1)]

            step_batch = tuple(slice_batch(x) for x in batch)
            rng, k1, k2 = jax.random.split(rng, 3)
            state, metrics = self._update_critic(state, step_batch, k1)

            if i in policy_update_indices:
                if self._aug:
                    state, actor_metrics = self._update_actor_aug(state, step_batch[0], step_batch[1], k2, entropy_alpha)
                else:
                    mem_start = actor_slot * step_batch[0].shape[0]
                    mem_end = mem_start + step_batch[0].shape[0]
                    state, best_actions, actor_metrics = self._update_actor_memory(
                        state,
                        memory_states[mem_start:mem_end],
                        memory_actions[mem_start:mem_end],
                        k2,
                    )
                    updated_memory_actions.append(best_actions)
                    actor_slot += 1
                actor_count = actor_count + 1.0
                actor_loss = actor_metrics["actor_loss"]
                q_weight_mean = actor_metrics["q_weight_mean"]
                q_selected_mean = actor_metrics["q_selected_mean"]
                q_selected_std = actor_metrics["q_selected_std"]
                running_q_mean = actor_metrics["running_q_mean"]
                running_q_std = actor_metrics["running_q_std"]
                action_gradient_loss = actor_metrics["action_gradient_loss"]
                action_gradient_norm = actor_metrics["action_gradient_norm"]
                entropy_alpha_metric = actor_metrics["entropy_alpha"]
            else:
                actor_loss = jnp.asarray(0.0)
                q_weight_mean = jnp.asarray(0.0)
                q_selected_mean = jnp.asarray(0.0)
                q_selected_std = jnp.asarray(0.0)
                running_q_mean = jnp.asarray(0.0)
                running_q_std = jnp.asarray(0.0)
                action_gradient_loss = jnp.asarray(0.0)
                action_gradient_norm = jnp.asarray(0.0)
                entropy_alpha_metric = jnp.asarray(0.0)

            if i in target_update_indices:
                state = state.replace(
                    target_actor_params=soft_update(
                        state.target_actor_params,
                        state.actor_params,
                        self.actor_tau,
                    )
                )

            metric_sums = {
                "critic_loss": metric_sums["critic_loss"] + metrics["critic_loss"],
                "current_q_values": metric_sums["current_q_values"] + metrics["current_q_values"],
                "next_q_values": metric_sums["next_q_values"] + metrics["next_q_values"],
                "target_q_values": metric_sums["target_q_values"] + metrics["target_q_values"],
                "q_std": metric_sums["q_std"] + metrics["q_std"],
                "actor_loss": metric_sums["actor_loss"] + actor_loss,
                "q_weight_mean": metric_sums["q_weight_mean"] + q_weight_mean,
                "q_selected_mean": metric_sums["q_selected_mean"] + q_selected_mean,
                "q_selected_std": metric_sums["q_selected_std"] + q_selected_std,
                "running_q_mean": metric_sums["running_q_mean"] + running_q_mean,
                "running_q_std": metric_sums["running_q_std"] + running_q_std,
                "action_gradient_loss": metric_sums["action_gradient_loss"] + action_gradient_loss,
                "action_gradient_norm": metric_sums["action_gradient_norm"] + action_gradient_norm,
                "entropy_alpha": metric_sums["entropy_alpha"] + entropy_alpha_metric,
            }

        critic_denom = jnp.asarray(float(gradient_steps))
        actor_denom = jnp.maximum(actor_count, 1.0)
        log_metrics = {
            "critic_loss": metric_sums["critic_loss"] / critic_denom,
            "current_q_values": metric_sums["current_q_values"] / critic_denom,
            "next_q_values": metric_sums["next_q_values"] / critic_denom,
            "target_q_values": metric_sums["target_q_values"] / critic_denom,
            "q_std": metric_sums["q_std"] / critic_denom,
            "actor_loss": metric_sums["actor_loss"] / actor_denom,
            "q_weight_mean": metric_sums["q_weight_mean"] / actor_denom,
            "q_selected_mean": metric_sums["q_selected_mean"] / actor_denom,
            "q_selected_std": metric_sums["q_selected_std"] / actor_denom,
            "running_q_mean": metric_sums["running_q_mean"] / actor_denom,
            "running_q_std": metric_sums["running_q_std"] / actor_denom,
            "action_gradient_loss": metric_sums["action_gradient_loss"] / actor_denom,
            "action_gradient_norm": metric_sums["action_gradient_norm"] / actor_denom,
            "entropy_alpha": metric_sums["entropy_alpha"] / actor_denom,
        }
        if updated_memory_actions:
            memory_actions_out = jnp.concatenate(updated_memory_actions, axis=0)
        else:
            memory_actions_out = memory_actions
        return state, rng, memory_actions_out, log_metrics

    def train(self, batch_size, gradient_steps):
        data = sample_replay_per_step(
            self.replay_buffer,
            batch_size,
            gradient_steps,
            env=self._vec_normalize_env,
            mix_recent=self.mix_recent_replay,
            recent_ratio=self.recent_replay_ratio,
            recent_window=self.recent_replay_window,
        )
        batch = self._prepare_batch(data)
        policy_update_indices = tuple(
            i
            for i in range(gradient_steps)
            if ((self._n_updates + i + 1) % self.policy_delay) == 0
        )
        target_update_indices = tuple(
            i for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.target_update_interval) == 0
        )
        num_actor_updates = len(policy_update_indices)
        if num_actor_updates and not self._aug:
            mem_idx, mem_states, mem_actions = self._sample_diffusion_memory(batch_size * num_actor_updates)
        else:
            assert self._diffusion_obs is not None
            mem_idx = np.empty((0,), dtype=np.int64)
            mem_states = jnp.zeros((0, self._diffusion_obs.shape[1]), dtype=jnp.float32)
            mem_actions = jnp.zeros((0, self._d_a), dtype=jnp.float32)

        self.policy.state, self.key, updated_mem_actions, log_metrics = self._train_fused(
            self.policy.state,
            batch,
            int(gradient_steps),
            policy_update_indices,
            target_update_indices,
            self.key,
            self._current_entropy_alpha(),
            mem_states,
            mem_actions,
        )
        if num_actor_updates and not self._aug:
            self._replace_diffusion_actions(mem_idx, updated_mem_actions)
        self._n_updates += gradient_steps

        metric_prefix = f"train/{self.cfg.alg.name}"
        self.logger.record(f"{metric_prefix}/n_updates", self._n_updates, exclude="tensorboard")
        for key, value in log_metrics.items():
            try:
                value = value.item()
            except Exception:
                pass
            self.logger.record(f"{metric_prefix}/{key}", value)
