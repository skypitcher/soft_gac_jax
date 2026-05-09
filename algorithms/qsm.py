"""QSM: Q-Score Matching (arXiv:2312.11752).

paper: https://arxiv.org/abs/2312.11752
code: https://github.com/escontra/score_matching_rl
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

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
from models.qsm_policy import QSMPolicy


def _sanitize_label_part(value: str) -> str:
    return str(value).replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


class QSM(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[QSMPolicy]]] = {
        "MlpPolicy": QSMPolicy,
        "MultiInputPolicy": QSMPolicy,
    }
    policy: QSMPolicy
    action_space: spaces.Box

    @staticmethod
    def default_run_label(cfg) -> str:
        domain, task = _env_parts(cfg)
        env_label = f"{domain}_{task}" if domain else task
        critic_hs = [int(h) for h in cfg.alg.critic.hs]
        critic_layers = len(critic_hs)
        critic_hidden = str(critic_hs[0]) if len(set(critic_hs)) == 1 else "-".join(str(h) for h in critic_hs)
        critic_label = f"cxq1_c{critic_hidden}x{critic_layers}"
        return (
            f"qsm_{env_label}_"
            f"a{int(cfg.alg.actor.hidden_size)}x{int(cfg.alg.actor.num_layers)}_"
            f"{critic_label}_"
            f"I{int(cfg.alg.actor.iter_steps)}_"
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
        self._T = int(cfg.alg.actor.iter_steps)
        self._M_q = float(cfg.alg.qsm.m_q)
        self.policy_delay = int(cfg.alg.train.policy_delay)
        self._ddpm_temperature = float(cfg.alg.qsm.ddpm_temperature)
        self._clip_sampler = bool(cfg.alg.qsm.clip_sampler)
        self._beta_schedule = str(cfg.alg.qsm.beta_schedule).lower()
        self._d_a = int(np.prod(self.action_space.shape))
        self.action_high = jnp.array(self.action_space.high, dtype=jnp.float32)
        self.action_low = jnp.array(self.action_space.low, dtype=jnp.float32)
        self._num_atoms = int(cfg.alg.critic.n_atoms)
        self._v_min = float(cfg.alg.critic.v_min)
        self._v_max = float(cfg.alg.critic.v_max)
        self._z_atoms = jnp.linspace(self._v_min, self._v_max, self._num_atoms)
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
            self.denoiser_module = self.policy.denoiser_module
            self.actor_optimizer = self.policy.actor_optimizer
            self._sqrt_alphas_cumprod = self.policy._sqrt_alphas_cumprod
            self._sqrt_one_minus_alphas_cumprod = self.policy._sqrt_one_minus_alphas_cumprod
            self._sqrt_recip_alphas = self.policy._sqrt_recip_alphas
            self._noise_coeff = self.policy._noise_coeff
            self._sqrt_betas = self.policy._sqrt_betas

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in self.cfg.alg.critic.hs]
        return {
            "family": "qsm",
            "actor_hidden_size": int(self.cfg.alg.actor.hidden_size),
            "actor_num_layers": int(self.cfg.alg.actor.num_layers),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "critic_crossq": True,
            "critic_update": "crossq_c51",
            "diffusion_steps": int(self._T),
            "policy_delay": int(self.policy_delay),
            "m_q": float(self._M_q),
            "beta_schedule": self._beta_schedule,
            "ddpm_temperature": float(self._ddpm_temperature),
            "clip_sampler": bool(self._clip_sampler),
            "mix_recent": bool(self.mix_recent_replay),
            "recent_ratio": float(self.recent_replay_ratio),
            "recent_window": int(self.recent_replay_window),
            "bn_momentum": float(self.cfg.alg.optimizer.bn_momentum),
            "bn_warmup": int(self.cfg.alg.optimizer.bn_warmup),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "actor_params": self._count_params(self.policy.state.actor_params),
            "critic_params": self._count_params(self.policy.state.qf_state.params),
        }

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

    def _forward_diffusion(self, a_0, t, noise):
        sqrt_alpha_bar = self._sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus = self._sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha_bar * a_0 + sqrt_one_minus * noise

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reverse_sample(self, actor_params, obs, rng):
        b = obs.shape[0]
        rng, init_key = jax.random.split(rng)
        x = jax.random.normal(init_key, (b, self._d_a))

        def reverse_step(carry, t_idx):
            x_curr, rng_carry = carry
            t_batch = jnp.full((b,), t_idx, dtype=jnp.int32)
            eps_pred = self.denoiser_module.apply(actor_params, x_curr, t_batch, obs)
            noise_c = self._noise_coeff[t_idx]
            recip_alpha = self._sqrt_recip_alphas[t_idx]
            mean = recip_alpha * (x_curr - noise_c * eps_pred)
            rng_carry, z_key = jax.random.split(rng_carry)
            z = jax.random.normal(z_key, x_curr.shape) * self._ddpm_temperature
            sigma = self._sqrt_betas[t_idx]
            x_next = mean + sigma * z * (t_idx > 0).astype(jnp.float32)
            if self._clip_sampler:
                x_next = jnp.clip(x_next, -1.0, 1.0)
            return (x_next, rng_carry), None

        t_steps = jnp.arange(self._T - 1, -1, -1)
        (x_final, _), _ = jax.lax.scan(reverse_step, (x, rng), t_steps)
        return jnp.clip(x_final, self.action_low, self.action_high)

    def _critic_eval(self, qf_state, obs, act):
        return qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            obs,
            act,
            train=False,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_critic(self, state, batch, rng):
        s, a, r, ns, dones = batch
        sample_key, noise_key = jax.random.split(rng)
        next_actions = self._reverse_sample(state.actor_params, ns, sample_key)
        smooth_noise = jax.random.normal(noise_key, next_actions.shape) * 0.1
        next_actions = jnp.clip(next_actions + smooth_noise, self.action_low, self.action_high)
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
        new_state = state.replace(
            qf_state=qf_state,
            critic_updates=state.critic_updates + 1,
        )
        return new_state, dict(metrics, q_std=current_q.std())

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_actor(self, state, states, actions, rng):
        k_t, k_eps = jax.random.split(rng)
        batch_size = states.shape[0]
        t = jax.random.randint(k_t, (batch_size,), 0, self._T)
        eps = jax.random.normal(k_eps, actions.shape)
        noisy_actions = self._forward_diffusion(actions, t, eps)

        def q1_fn(a):
            q_dist = self._critic_eval(state.qf_state, states, a)
            q1, _, _ = twin_q_expectation(q_dist, self._z_atoms)
            return q1.sum()

        def q2_fn(a):
            q_dist = self._critic_eval(state.qf_state, states, a)
            _, q2, _ = twin_q_expectation(q_dist, self._z_atoms)
            return q2.sum()

        critic_jacobian = jax.lax.stop_gradient(0.5 * (jax.grad(q1_fn)(noisy_actions) + jax.grad(q2_fn)(noisy_actions)))

        def actor_loss_fn(actor_params):
            pred = self.denoiser_module.apply(actor_params, noisy_actions, t, states)
            return jnp.mean((-self._M_q * critic_jacobian - pred) ** 2)

        loss, grads = jax.value_and_grad(actor_loss_fn)(state.actor_params)
        updates, new_opt = self.actor_optimizer.update(grads, state.actor_opt_state, state.actor_params)
        new_actor_params = optax.apply_updates(state.actor_params, updates)
        new_state = state.replace(
            actor_params=new_actor_params,
            actor_opt_state=new_opt,
            actor_updates=state.actor_updates + 1,
        )
        return new_state, {"actor_loss": loss}

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def _train_fused(self, state, batch, gradient_steps: int, n_updates, rng):
        metric_sums = {
            "critic_loss": jnp.asarray(0.0),
            "current_q_values": jnp.asarray(0.0),
            "next_q_values": jnp.asarray(0.0),
            "target_q_values": jnp.asarray(0.0),
            "q_std": jnp.asarray(0.0),
            "actor_loss": jnp.asarray(0.0),
        }
        actor_count = jnp.asarray(0.0)

        for i in range(gradient_steps):
            def slice_batch(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                step_batch_size = x.shape[0] // gradient_steps
                return x[step_batch_size * step: step_batch_size * (step + 1)]

            step_batch = tuple(slice_batch(x) for x in batch)
            rng, k1, k2 = jax.random.split(rng, 3)
            state, metrics = self._update_critic(state, step_batch, k1)
            update_idx = n_updates + i + 1

            def update_actor(st):
                new_state, actor_metrics = self._update_actor(st, step_batch[0], step_batch[1], k2)
                return new_state, actor_metrics["actor_loss"], jnp.asarray(1.0)

            state, actor_loss, actor_updated = jax.lax.cond(
                (update_idx % self.policy_delay) == 0,
                update_actor,
                lambda st: (st, jnp.asarray(0.0), jnp.asarray(0.0)),
                state,
            )
            actor_count = actor_count + actor_updated

            metric_sums = {
                "critic_loss": metric_sums["critic_loss"] + metrics["critic_loss"],
                "current_q_values": metric_sums["current_q_values"] + metrics["current_q_values"],
                "next_q_values": metric_sums["next_q_values"] + metrics["next_q_values"],
                "target_q_values": metric_sums["target_q_values"] + metrics["target_q_values"],
                "q_std": metric_sums["q_std"] + metrics["q_std"],
                "actor_loss": metric_sums["actor_loss"] + actor_loss,
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
        }
        return state, rng, log_metrics

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
        self.policy.state, self.key, log_metrics = self._train_fused(
            self.policy.state,
            batch,
            int(gradient_steps),
            jnp.asarray(self._n_updates),
            self.key,
        )
        self._n_updates += gradient_steps

        metric_prefix = f"train/{self.cfg.alg.name}"
        self.logger.record(f"{metric_prefix}/n_updates", self._n_updates, exclude="tensorboard")
        for key, value in log_metrics.items():
            try:
                value = value.item()
            except Exception:
                pass
            self.logger.record(f"{metric_prefix}/{key}", value)
