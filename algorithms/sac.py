from __future__ import annotations

import functools
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv

from common.buffers import sample_mixed_recent, sample_replay_per_step
from common.crossq_c51 import twin_q_expectation, update_crossq_c51_critic
from common.label_utils import replay_mix_recent_label
from common.off_policy_algorithm import OffPolicyAlgorithmJax
from models.sac_policy import SACPolicy


def _sanitize_label_part(value: str) -> str:
    return str(value).replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {
        "MlpPolicy": SACPolicy,
        "MultiInputPolicy": SACPolicy,
    }
    policy: SACPolicy
    action_space: spaces.Box

    @staticmethod
    def default_run_label(cfg) -> str:
        domain, task = _env_parts(cfg)
        env_label = f"{domain}_{task}" if domain else task
        family_label = "crossq_c51_sac"
        critic_hs = [int(h) for h in cfg.alg.critic.hs]
        critic_layers = len(critic_hs)
        critic_hidden = str(critic_hs[0]) if len(set(critic_hs)) == 1 else "-".join(str(h) for h in critic_hs)
        critic_label = f"crossq_c{critic_hidden}x{critic_layers}_z{int(cfg.alg.critic.n_atoms)}"
        alpha_cfg = cfg.alg.alpha
        alpha_label = "auto" if not isinstance(alpha_cfg, (int, float)) and str(alpha_cfg.get("type", "auto")) == "auto" else "const"
        return (
            f"{family_label}_{env_label}_"
            f"a{int(cfg.alg.actor.hidden_size)}x{int(cfg.alg.actor.num_layers)}_"
            f"{critic_label}_"
            f"utd{int(cfg.alg.train.utd)}_"
            f"pd{int(cfg.alg.train.policy_delay)}_"
            f"alpha{alpha_label}_"
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
        self._num_atoms = int(cfg.alg.critic.n_atoms)
        if self._num_atoms < 2:
            raise ValueError("SAC uses the shared CrossQ+C51 critic and requires alg.critic.n_atoms >= 2.")
        self._v_min = float(cfg.alg.critic.v_min)
        self._v_max = float(cfg.alg.critic.v_max)
        self._z_atoms = jnp.linspace(self._v_min, self._v_max, self._num_atoms)
        self._auto_alpha = self._alpha_type() == "auto"
        self._target_entropy = self._resolve_target_entropy()
        if _init_setup_model:
            self._setup_model()

    def _alpha_type(self) -> str:
        alpha_cfg = self.cfg.alg.alpha
        if isinstance(alpha_cfg, (int, float)):
            return "const"
        return str(alpha_cfg.get("type", "auto"))

    def _resolve_target_entropy(self) -> float:
        alpha_cfg = self.cfg.alg.alpha
        action_dim = int(np.prod(self.action_space.shape))
        if isinstance(alpha_cfg, (int, float)):
            return -float(action_dim)
        raw = alpha_cfg.get("target_entropy", "auto")
        if str(raw) == "auto":
            return -float(action_dim)
        return float(raw)

    def _setup_model(self, reset: bool = False) -> None:
        if not reset:
            super()._setup_model()
        if not hasattr(self, "policy") or self.policy is None or reset:
            self.policy = self.policy_class(self.observation_space, self.action_space, self.cfg)
            assert isinstance(self.qf_learning_rate, float)
            self.key, _ = jax.random.split(self.key)
            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)
            self.actor_module = self.policy.actor_module
            self.actor_optimizer = self.policy.actor_optimizer
            self.alpha_optimizer = self.policy.alpha_optimizer

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in self.cfg.alg.critic.hs]
        alpha_cfg = self.cfg.alg.alpha
        alpha_init = float(alpha_cfg) if isinstance(alpha_cfg, (int, float)) else float(alpha_cfg.init)
        alpha_type = self._alpha_type()
        alpha_lr = 0.0 if alpha_type != "auto" else float(alpha_cfg.get("lr", 0.0))
        return {
            "family": "crossq_c51_sac",
            "actor_hidden_size": int(self.cfg.alg.actor.hidden_size),
            "actor_num_layers": int(self.cfg.alg.actor.num_layers),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "critic_crossq": True,
            "critic_update": "crossq_c51",
            "critic_n_atoms": int(self._num_atoms),
            "policy_delay": int(self.policy_delay),
            "mix_recent": bool(self.mix_recent_replay),
            "recent_ratio": float(self.recent_replay_ratio),
            "recent_window": int(self.recent_replay_window),
            "bn": bool(self.cfg.alg.optimizer.bn),
            "bn_mode": str(self.cfg.alg.optimizer.bn_mode),
            "bn_momentum": float(self.cfg.alg.optimizer.bn_momentum),
            "bn_warmup": int(self.cfg.alg.optimizer.bn_warmup),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "alpha_type": alpha_type,
            "alpha_init": alpha_init,
            "alpha_lr": alpha_lr,
            "target_entropy": float(self._target_entropy),
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

    def _critic_eval(self, qf_state, obs, act, rng):
        return qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            obs,
            act,
            rngs={"dropout": rng},
            train=False,
        )

    def _sample_actor(self, actor_params, actor_batch_stats, obs, rng):
        return self.actor_module.apply(
            {"params": actor_params, "batch_stats": actor_batch_stats},
            obs,
            rng=rng,
            method=self.actor_module.sample_and_log_prob,
            train=False,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_critic(self, state, batch, rng):
        s, a, r, ns, dones = batch
        alpha = jnp.exp(state.log_alpha).reshape(())
        rng, actor_rng = jax.random.split(rng)
        next_action, next_log_prob = self._sample_actor(state.actor_params, state.actor_batch_stats, ns, actor_rng)
        next_penalty = alpha * next_log_prob
        qf_state, metrics, rng, current_q = update_crossq_c51_critic(
            self.gamma,
            state.qf_state,
            s,
            a,
            ns,
            next_action,
            r,
            dones,
            next_penalty,
            self._num_atoms,
            self._z_atoms,
            self._v_min,
            self._v_max,
            float(self.cfg.alg.critic.dist_entropy_coeff),
            rng,
        )
        metrics = dict(metrics, target_q_values=metrics["next_q_values"])
        return state.replace(
            qf_state=qf_state,
            critic_updates=state.critic_updates + 1,
        ), dict(metrics, q_std=current_q.std(), next_log_prob=next_log_prob.mean())

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_actor_and_alpha(self, state, obs, rng):
        alpha = jnp.exp(state.log_alpha).reshape(())
        sample_rng, dropout_rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            (action, log_prob), state_updates = self.actor_module.apply(
                {"params": actor_params, "batch_stats": state.actor_batch_stats},
                obs,
                rng=sample_rng,
                method=self.actor_module.sample_and_log_prob,
                mutable=["batch_stats"],
                train=True,
            )
            q_dist = self._critic_eval(state.qf_state, obs, action, dropout_rng)
            _, _, min_q = twin_q_expectation(q_dist, self._z_atoms)
            min_q = min_q.reshape(-1, 1)
            loss = (jax.lax.stop_gradient(alpha) * log_prob - min_q).mean()
            return loss, (log_prob.mean(), min_q.mean(), state_updates.get("batch_stats", state.actor_batch_stats))

        (actor_loss, (log_prob_mean, q_mean, actor_batch_stats)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
        actor_updates, actor_opt_state = self.actor_optimizer.update(grads, state.actor_opt_state, state.actor_params)
        actor_params = jax.tree_util.tree_map(lambda p, u: p + u, state.actor_params, actor_updates)

        if self._auto_alpha:
            def alpha_loss_fn(log_alpha):
                return -(log_alpha * jax.lax.stop_gradient(log_prob_mean + self._target_entropy)).mean()

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)
            alpha_updates, alpha_opt_state = self.alpha_optimizer.update(alpha_grads, state.alpha_opt_state, state.log_alpha)
            log_alpha = jax.tree_util.tree_map(lambda p, u: p + u, state.log_alpha, alpha_updates)
        else:
            alpha_loss = jnp.array(0.0)
            alpha_opt_state = state.alpha_opt_state
            log_alpha = state.log_alpha

        return state.replace(
            actor_params=actor_params,
            actor_batch_stats=actor_batch_stats,
            actor_opt_state=actor_opt_state,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            actor_updates=state.actor_updates + 1,
        ), {
            "actor_loss": actor_loss,
            "actor_log_prob": log_prob_mean,
            "actor_q": q_mean,
            "alpha": jnp.exp(log_alpha).squeeze(),
            "alpha_loss": alpha_loss,
        }

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def _train_fused(self, state, batch, gradient_steps: int, n_updates, rng):
        metric_sums = {
            "critic_loss": jnp.asarray(0.0),
            "current_q_values": jnp.asarray(0.0),
            "next_q_values": jnp.asarray(0.0),
            "target_q_values": jnp.asarray(0.0),
            "q_std": jnp.asarray(0.0),
            "next_log_prob": jnp.asarray(0.0),
            "actor_loss": jnp.asarray(0.0),
            "actor_log_prob": jnp.asarray(0.0),
            "actor_q": jnp.asarray(0.0),
            "alpha": jnp.asarray(0.0),
            "alpha_loss": jnp.asarray(0.0),
        }
        actor_count = jnp.asarray(0.0)

        for i in range(gradient_steps):
            def slice_batch(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                step_batch_size = x.shape[0] // gradient_steps
                return x[step_batch_size * step: step_batch_size * (step + 1)]

            step_batch = tuple(slice_batch(x) for x in batch)
            rng, critic_key, actor_key = jax.random.split(rng, 3)
            state, metrics = self._update_critic(state, step_batch, critic_key)
            update_idx = n_updates + i + 1

            def update_actor(st):
                new_state, actor_metrics = self._update_actor_and_alpha(st, step_batch[0], actor_key)
                return (
                    new_state,
                    actor_metrics["actor_loss"],
                    actor_metrics["actor_log_prob"],
                    actor_metrics["actor_q"],
                    actor_metrics["alpha"],
                    actor_metrics["alpha_loss"],
                    jnp.asarray(1.0),
                )

            state, actor_loss, actor_log_prob, actor_q, alpha, alpha_loss, actor_updated = jax.lax.cond(
                (update_idx % self.policy_delay) == 0,
                update_actor,
                lambda st: (
                    st,
                    jnp.asarray(0.0),
                    jnp.asarray(0.0),
                    jnp.asarray(0.0),
                    jnp.exp(st.log_alpha).squeeze(),
                    jnp.asarray(0.0),
                    jnp.asarray(0.0),
                ),
                state,
            )
            actor_count = actor_count + actor_updated

            metric_sums = {
                "critic_loss": metric_sums["critic_loss"] + metrics["critic_loss"],
                "current_q_values": metric_sums["current_q_values"] + metrics["current_q_values"],
                "next_q_values": metric_sums["next_q_values"] + metrics["next_q_values"],
                "target_q_values": metric_sums["target_q_values"] + metrics["target_q_values"],
                "q_std": metric_sums["q_std"] + metrics["q_std"],
                "next_log_prob": metric_sums["next_log_prob"] + metrics["next_log_prob"],
                "actor_loss": metric_sums["actor_loss"] + actor_loss,
                "actor_log_prob": metric_sums["actor_log_prob"] + actor_log_prob,
                "actor_q": metric_sums["actor_q"] + actor_q,
                "alpha": metric_sums["alpha"] + alpha,
                "alpha_loss": metric_sums["alpha_loss"] + alpha_loss,
            }

        critic_denom = jnp.asarray(float(gradient_steps))
        actor_denom = jnp.maximum(actor_count, 1.0)
        log_metrics = {
            "critic_loss": metric_sums["critic_loss"] / critic_denom,
            "current_q_values": metric_sums["current_q_values"] / critic_denom,
            "next_q_values": metric_sums["next_q_values"] / critic_denom,
            "target_q_values": metric_sums["target_q_values"] / critic_denom,
            "q_std": metric_sums["q_std"] / critic_denom,
            "next_log_prob": metric_sums["next_log_prob"] / critic_denom,
            "actor_loss": metric_sums["actor_loss"] / actor_denom,
            "actor_log_prob": metric_sums["actor_log_prob"] / actor_denom,
            "actor_q": metric_sums["actor_q"] / actor_denom,
            "alpha": metric_sums["alpha"] / actor_denom,
            "alpha_loss": metric_sums["alpha_loss"] / actor_denom,
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
