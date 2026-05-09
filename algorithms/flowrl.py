"""FlowRL baseline on the shared SB3Jax training stack.

paper: https://arxiv.org/pdf/2506.12811
code: https://github.com/bytedance/FlowRL
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
from models.flowrl_policy import FlowRLPolicy
from models.legacy_utils import soft_update


def _sanitize_label_part(value: str) -> str:
    return str(value).replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


class FlowRL(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[FlowRLPolicy]]] = {
        "MlpPolicy": FlowRLPolicy,
        "MultiInputPolicy": FlowRLPolicy,
    }
    policy: FlowRLPolicy
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
            f"flowrl_{env_label}_"
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
        self.policy_delay = int(cfg.alg.train.policy_delay)
        self.target_update_interval = int(cfg.alg.target.target_update_interval)
        self._quantile = float(cfg.alg.flowrl.quantile)
        self._lamda = float(cfg.alg.flowrl.lamda)
        self._cfm_min = float(cfg.alg.flowrl.cfm_min)
        self._cfm_max = float(cfg.alg.flowrl.cfm_max)
        self._polyak_weight = float(cfg.alg.target.polyak_weight)
        self._actor_iter_steps = int(cfg.alg.actor.iter_steps)
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
            self.critic_buf_module = self.policy.critic_buf_module
            self.v_buf_module = self.policy.v_buf_module
            self.policy_module = self.policy.policy_module
            self.policy_optimizer = self.policy.policy_optimizer
            self.critic_buf_optimizer = self.policy.critic_buf_optimizer
            self.v_buf_optimizer = self.policy.v_buf_optimizer

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in self.cfg.alg.critic.hs]
        main_critic = self._count_params(self.policy.state.qf_state.params)
        buf_critic = self._count_params(self.policy.state.critic_buf_params)
        value_buf = self._count_params(self.policy.state.v_buf_params)
        return {
            "family": "flowrl",
            "actor_hidden_size": int(self.cfg.alg.actor.hidden_size),
            "actor_num_layers": int(self.cfg.alg.actor.num_layers),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "critic_crossq": True,
            "critic_update": "crossq_c51",
            "integration_steps": int(self._actor_iter_steps),
            "policy_delay": int(self.policy_delay),
            "mix_recent": bool(self.mix_recent_replay),
            "recent_ratio": float(self.recent_replay_ratio),
            "recent_window": int(self.recent_replay_window),
            "bn_momentum": float(self.cfg.alg.optimizer.bn_momentum),
            "bn_warmup": int(self.cfg.alg.optimizer.bn_warmup),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "buffer_target_update_interval": int(self.target_update_interval),
            "quantile": float(self._quantile),
            "lamda": float(self._lamda),
            "cfm_min": float(self._cfm_min),
            "cfm_max": float(self._cfm_max),
            "polyak_weight": float(self._polyak_weight),
            "buffer_hidden_size": int(self.cfg.alg.flowrl.buffer_hidden_size),
            "buffer_num_layers": int(self.cfg.alg.flowrl.buffer_num_layers),
            "actor_params": self._count_params(self.policy.state.policy_params),
            "critic_params": main_critic + buf_critic + value_buf,
            "critic_main_params": main_critic,
            "critic_buffer_params": buf_critic,
            "value_buffer_params": value_buf,
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

    def _critic_eval(self, qf_state, obs, act):
        return qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            obs,
            act,
            train=False,
        )

    def _critic_buf_eval(self, params, obs, act):
        return self.critic_buf_module.apply({"params": params}, obs, act)

    def _v_buf_eval(self, params, obs):
        return self.v_buf_module.apply({"params": params}, obs)

    def _policy_sample(self, params, obs, rng):
        return self.policy_module.apply({"params": params}, obs, rng=rng, method=self.policy_module.sample, train=False)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_critic(self, state, batch, rng):
        s, a, r, ns, dones = batch
        next_action = self._policy_sample(state.policy_params, ns, rng)
        zero_penalty = jnp.zeros((s.shape[0],), dtype=jnp.float32)
        qf_state, metrics, rng, _ = update_crossq_c51_critic(
            self.gamma,
            state.qf_state,
            s,
            a,
            ns,
            next_action,
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
        mask = 1.0 - dones
        target_vf = self._v_buf_eval(state.v_buf_params, ns)
        target_q_buf = jax.lax.stop_gradient(r + mask * self.gamma * target_vf)

        def buf_loss_fn(params):
            q1b, q2b = self._critic_buf_eval(params, s, a)
            q_buffer_ = jnp.minimum(q1b, q2b)
            loss = jnp.mean((q1b - target_q_buf) ** 2) + jnp.mean((q2b - target_q_buf) ** 2)
            return loss, q_buffer_

        (buffer_loss, q_buffer), b_grads = jax.value_and_grad(buf_loss_fn, has_aux=True)(state.critic_buf_params)
        b_updates, new_b_opt = self.critic_buf_optimizer.update(
            b_grads,
            state.critic_buf_opt_state,
            state.critic_buf_params,
        )
        new_b_params = optax.apply_updates(state.critic_buf_params, b_updates)

        q_pred = jax.lax.stop_gradient(jnp.minimum(*self._critic_buf_eval(state.critic_target_buf_params, s, a)))

        def vf_loss_fn(params):
            vf_pred = self._v_buf_eval(params, s)
            vf_err = q_pred - vf_pred
            vf_sign = (vf_err < 0).astype(jnp.float32)
            vf_weight = (1.0 - vf_sign) * self._quantile + vf_sign * (1.0 - self._quantile)
            return jnp.mean(vf_weight * (vf_err ** 2))

        value_loss, v_grads = jax.value_and_grad(vf_loss_fn)(state.v_buf_params)
        v_updates, new_v_opt = self.v_buf_optimizer.update(v_grads, state.v_buf_opt_state, state.v_buf_params)
        new_v_params = optax.apply_updates(state.v_buf_params, v_updates)

        new_state = state.replace(
            qf_state=qf_state,
            critic_buf_params=new_b_params,
            critic_buf_opt_state=new_b_opt,
            v_buf_params=new_v_params,
            v_buf_opt_state=new_v_opt,
            critic_updates=state.critic_updates + 1,
        )
        metrics = dict(
            metrics,
            buffer_critic_loss=buffer_loss,
            value_buffer_loss=value_loss,
            q_buffer_mean=q_buffer.mean(),
            q_buffer_std=q_buffer.std(),
        )
        return new_state, q_buffer, metrics

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_policy(self, state, s, a, q_buffer, rng):
        k1, k2 = jax.random.split(rng)
        action_0 = jnp.clip(jax.random.normal(k1, a.shape), -1, 1)

        def policy_loss_fn(policy_params):
            pi = self.policy_module.apply({"params": policy_params}, s, rng=k2, method=self.policy_module.sample, train=True)
            q_dist = self._critic_eval(state.qf_state, s, pi)
            _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
            min_q = min_q.reshape(-1, 1)
            weights = jnp.maximum(q_buffer - min_q, 0.0)
            weights = jnp.exp(weights - weights.mean())
            weights = jnp.clip(self._lamda * weights, self._cfm_min, self._cfm_max)
            weights = jax.lax.stop_gradient(weights)
            velocity_field = a - action_0
            t = jax.random.uniform(k1, (a.shape[0], 1))
            action_t = t * a + (1.0 - t) * action_0
            pred_v = self.policy_module.apply({"params": policy_params}, s, action_t, t, train=True)
            cfm_loss = weights * jnp.mean((pred_v - velocity_field) ** 2, axis=-1, keepdims=True)
            return (-min_q + cfm_loss).mean()

        loss, grads = jax.value_and_grad(policy_loss_fn)(state.policy_params)
        updates, new_opt = self.policy_optimizer.update(grads, state.policy_opt_state, state.policy_params)
        new_params = optax.apply_updates(state.policy_params, updates)
        return state.replace(
            policy_params=new_params,
            policy_opt_state=new_opt,
            policy_updates=state.policy_updates + 1,
        ), {"actor_loss": loss}

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def _train_fused(self, state, batch, gradient_steps: int, n_updates, rng):
        metric_sums = {
            "critic_loss": jnp.asarray(0.0),
            "buffer_critic_loss": jnp.asarray(0.0),
            "value_buffer_loss": jnp.asarray(0.0),
            "q_buffer_mean": jnp.asarray(0.0),
            "q_buffer_std": jnp.asarray(0.0),
            "current_q_values": jnp.asarray(0.0),
            "next_q_values": jnp.asarray(0.0),
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
            state, q_buffer, metrics = self._update_critic(state, step_batch, k1)
            update_idx = n_updates + i + 1

            def update_target(st):
                return st.replace(
                    critic_target_buf_params=soft_update(
                        st.critic_target_buf_params,
                        st.critic_buf_params,
                        self._polyak_weight,
                    ),
                )

            state = jax.lax.cond(
                (update_idx % self.target_update_interval) == 0,
                update_target,
                lambda st: st,
                state,
            )

            def update_actor(st):
                new_state, actor_metrics = self._update_policy(st, step_batch[0], step_batch[1], q_buffer, k2)
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
                "buffer_critic_loss": metric_sums["buffer_critic_loss"] + metrics["buffer_critic_loss"],
                "value_buffer_loss": metric_sums["value_buffer_loss"] + metrics["value_buffer_loss"],
                "q_buffer_mean": metric_sums["q_buffer_mean"] + metrics["q_buffer_mean"],
                "q_buffer_std": metric_sums["q_buffer_std"] + metrics["q_buffer_std"],
                "current_q_values": metric_sums["current_q_values"] + metrics["current_q_values"],
                "next_q_values": metric_sums["next_q_values"] + metrics["next_q_values"],
                "actor_loss": metric_sums["actor_loss"] + actor_loss,
            }

        denom = jnp.asarray(float(gradient_steps))
        log_metrics = {
            "critic_loss": metric_sums["critic_loss"] / denom,
            "buffer_critic_loss": metric_sums["buffer_critic_loss"] / denom,
            "value_buffer_loss": metric_sums["value_buffer_loss"] / denom,
            "q_buffer_mean": metric_sums["q_buffer_mean"] / denom,
            "q_buffer_std": metric_sums["q_buffer_std"] / denom,
            "current_q_values": metric_sums["current_q_values"] / denom,
            "next_q_values": metric_sums["next_q_values"] / denom,
            "actor_loss": metric_sums["actor_loss"] / jnp.maximum(actor_count, 1.0),
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
