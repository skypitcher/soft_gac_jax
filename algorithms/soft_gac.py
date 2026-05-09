import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from common.buffers import sample_replay_per_step
from common.config import config_to_plain_dict
from common.config_utils import cfg_get
from common.crossq_c51 import twin_q_expectation, update_crossq_c51_critic
from common.label_utils import replay_mix_recent_label
from common.off_policy_algorithm import OffPolicyAlgorithmJax
from common.soft_gac_base import base_distribution_label, canonical_base_distribution
from common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from models.soft_gac_policy import SoftGACPolicy


def _sanitize_label_part(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


def _alpha_label(cfg) -> str:
    alpha_cfg = cfg.alg.alpha
    if isinstance(alpha_cfg, (int, float)):
        return f"_alpha{float(alpha_cfg):g}"

    alpha_type = str(alpha_cfg.get("type", "auto"))
    if alpha_type == "auto":
        return ""
    if alpha_type == "const":
        return f"_alpha{float(alpha_cfg.get('init', 0.0)):g}"
    return f"_alpha{_sanitize_label_part(alpha_type)}"


def _canonical_particle_scheme(value: str | None) -> str:
    scheme = str(value or "random").strip().lower().replace("-", "_")
    if scheme in {"random", "iid", "mc", "standard"}:
        return "random"
    if scheme in {"antithetic", "ant", "mirror", "mirrored"}:
        return "antithetic"
    raise ValueError(f"Unsupported SoftGAC particle_scheme: {value}")


def _particle_scheme_label(cfg) -> str:
    scheme = _canonical_particle_scheme(
        cfg_get(cfg, "alg.train.particle_scheme", "alg.particle_scheme", default="random")
    )
    return "ant" if scheme == "antithetic" else ""


class TemperatureCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        del step
        log_alpha = self.param("log_alpha", init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_alpha)


class ConstantTemperatureCoef(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        del step
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.init_value))
        return jax.lax.stop_gradient(self.init_value)


class SoftGAC(OffPolicyAlgorithmJax):
    @staticmethod
    def default_run_label(cfg) -> str:
        domain, task = _env_parts(cfg)
        env_label = f"{domain}_{task}" if domain else task
        actor_hidden = int(cfg.alg.actor.hidden_size)
        actor_layers = int(cfg.alg.actor.num_layers)
        num_particles = int(cfg_get(cfg, "alg.train.num_particles", "alg.num_particles", default=8))

        critic_hs = [int(h) for h in cfg.alg.critic.hs]
        critic_layers = len(critic_hs)
        if len(set(critic_hs)) == 1:
            critic_hidden = str(critic_hs[0])
        else:
            critic_hidden = "-".join(str(h) for h in critic_hs)
        critic_label = f"cxq1_c{critic_hidden}x{critic_layers}"

        target_control_cfg = cfg.alg.target_control_energy
        if isinstance(target_control_cfg, (int, float)):
            rho_ctrl = float(target_control_cfg)
        elif str(target_control_cfg["type"]) == "auto":
            rho_ctrl = float(target_control_cfg["per_dim"])
        else:
            rho_ctrl = float(target_control_cfg["value"])

        return (
            f"soft_gac_{env_label}_"
            f"a{actor_hidden}x{actor_layers}_"
            f"{critic_label}_"
            f"rho{rho_ctrl:.2f}_"
            f"np{num_particles}{_particle_scheme_label(cfg)}_"
            f"{base_distribution_label(cfg_get(cfg, 'alg.actor.base_distribution', default='logistic'))}_"
            f"{replay_mix_recent_label(cfg)}"
            f"{_alpha_label(cfg)}"
        )

    policy_aliases: ClassVar[Dict[str, Type[SoftGACPolicy]]] = {
        "MlpPolicy": SoftGACPolicy,
        "MultiInputPolicy": SoftGACPolicy,
    }

    policy: SoftGACPolicy
    action_space: spaces.Box

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
        if cfg_get(cfg, "alg.critic.crossq", default=True) is False:
            raise ValueError("SoftGAC only supports the shared CrossQ+C51 critic. Remove alg.critic.crossq=false.")
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=cfg.alg.optimizer.lr_actor,
            qf_learning_rate=cfg.alg.optimizer.lr_critic,
            buffer_size=cfg_get(cfg, "alg.replay.buffer_size", "alg.buffer_size"),
            learning_starts=cfg_get(cfg, "alg.train.learning_starts", "alg.learning_starts"),
            batch_size=cfg_get(cfg, "alg.train.batch_size", "alg.batch_size"),
            tau=1.0,
            gamma=cfg_get(cfg, "alg.train.gamma", "alg.gamma"),
            train_freq=train_freq,
            gradient_steps=cfg_get(cfg, "alg.train.utd", "alg.utd"),
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=cfg.seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )
        self.cfg = cfg
        self.policy_delay = int(cfg.alg.train.policy_delay)
        self.policy_tau = float(cfg.alg.target.actor_tau)
        self.critic_use_batch_norm = bool(cfg_get(cfg, "alg.optimizer.bn", default=False))
        self.critic_use_layer_norm = bool(cfg_get(cfg, "alg.critic.use_layer_norm", default=False))
        self.mix_recent_replay = bool(cfg_get(cfg, "alg.replay.mix_recent", "alg.mix_recent_replay", default=True))
        self.recent_replay_ratio = float(cfg_get(cfg, "alg.replay.recent_ratio", "alg.recent_replay_ratio", default=0.2))
        self.recent_replay_window = int(cfg_get(cfg, "alg.replay.recent_window", "alg.recent_replay_window", default=4096))
        self.bridge_steps = int(cfg_get(cfg, "alg.actor.num_layers", "alg.actor_num_layers"))
        self.actor_base_distribution = canonical_base_distribution(
            cfg_get(cfg, "alg.actor.base_distribution", default="logistic")
        )
        self.num_particles = int(cfg_get(cfg, "alg.train.num_particles", "alg.num_particles"))
        self.particle_scheme = _canonical_particle_scheme(
            cfg_get(cfg, "alg.train.particle_scheme", "alg.particle_scheme", default="random")
        )
        self.action_dim = int(np.prod(self.action_space.shape))
        self.actor_noise_steps = self.bridge_steps + 1
        self.alpha_cfg = cfg.alg.alpha
        target_control_cfg = cfg.alg.target_control_energy
        if isinstance(target_control_cfg, (int, float)):
            self.target_control_energy = float(target_control_cfg)
        elif str(target_control_cfg["type"]) == "auto":
            self.target_control_energy = float(target_control_cfg["per_dim"]) * self.bridge_steps * self.action_dim
        elif str(target_control_cfg["type"]) == "const":
            self.target_control_energy = float(target_control_cfg["value"])
        else:
            raise NotImplementedError(f"Target control energy type {target_control_cfg['type']} not supported")
        self.model_save_path = model_save_path
        self.save_every_n_steps = save_every_n_steps
        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _write_flax_state(path: Path, state) -> None:
        path.write_bytes(flax.serialization.to_bytes(state))

    @staticmethod
    def _read_flax_state(path: Path, template):
        return flax.serialization.from_bytes(template, path.read_bytes())

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("step_"))
        except ValueError:
            return -1

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        keep_last: int | None = None,
        reason: str = "manual",
    ) -> str:
        checkpoint_root = Path(path or self.model_save_path or "checkpoint")
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        step = int(self.num_timesteps)
        step_name = f"step_{step:010d}"
        step_dir = checkpoint_root / step_name
        tmp_dir = checkpoint_root / f".{step_name}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        self._write_flax_state(tmp_dir / "actor_state.msgpack", self.policy.actor_state)
        self._write_flax_state(tmp_dir / "target_actor_state.msgpack", self.policy.target_actor_state)
        self._write_flax_state(tmp_dir / "qf_state.msgpack", self.policy.qf_state)
        self._write_flax_state(tmp_dir / "alpha_state.msgpack", self.alpha_state)
        np.save(tmp_dir / "key_data.npy", np.asarray(jax.random.key_data(self.key)))

        metadata = {
            "format_version": 1,
            "algorithm": str(self.cfg.alg.name),
            "env_name": str(self.cfg.env_name),
            "seed": int(self.cfg.seed),
            "num_timesteps": step,
            "n_updates": int(self._n_updates),
            "reason": reason,
            "actor_noise_steps": int(self.actor_noise_steps),
            "action_dim": int(self.action_dim),
            "use_crossq": True,
            "config": config_to_plain_dict(self.cfg),
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        if step_dir.exists():
            shutil.rmtree(step_dir)
        tmp_dir.rename(step_dir)
        (checkpoint_root / "latest.json").write_text(
            json.dumps({"path": step_name, "num_timesteps": step, "reason": reason}, indent=2),
            encoding="utf-8",
        )

        if keep_last is not None and keep_last > 0:
            step_dirs = sorted(
                (p for p in checkpoint_root.glob("step_*") if p.is_dir()),
                key=self._checkpoint_step,
            )
            for old_dir in step_dirs[:-keep_last]:
                shutil.rmtree(old_dir, ignore_errors=True)

        return str(step_dir)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path)
        if (checkpoint_path / "latest.json").is_file():
            latest = json.loads((checkpoint_path / "latest.json").read_text(encoding="utf-8"))
            checkpoint_path = checkpoint_path / latest["path"]

        self.policy.actor_state = self._read_flax_state(checkpoint_path / "actor_state.msgpack", self.policy.actor_state)
        target_actor_path = checkpoint_path / "target_actor_state.msgpack"
        if target_actor_path.is_file():
            self.policy.target_actor_state = self._read_flax_state(target_actor_path, self.policy.target_actor_state)
        qf_path = checkpoint_path / "qf_state.msgpack"
        if qf_path.is_file():
            self.policy.qf_state = self._read_flax_state(qf_path, self.policy.qf_state)
        alpha_path = checkpoint_path / "alpha_state.msgpack"
        if alpha_path.is_file():
            self.alpha_state = self._read_flax_state(alpha_path, self.alpha_state)
        key_path = checkpoint_path / "key_data.npy"
        if key_path.is_file():
            self.key = jax.random.wrap_key_data(jnp.asarray(np.load(key_path)))

        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {}

    def _setup_model(self, reset: bool = False) -> None:
        if not reset:
            super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None or reset:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.cfg,
            )
            assert isinstance(self.qf_learning_rate, float)
            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)
            self.key, alpha_key = jax.random.split(self.key, 2)

            if isinstance(self.alpha_cfg, (int, float)):
                self.alpha_module = ConstantTemperatureCoef(float(self.alpha_cfg))
                alpha_lr = 0.0
            else:
                alpha_type = str(self.alpha_cfg["type"])
                alpha_init = float(self.alpha_cfg["init"])
                if alpha_type == "auto":
                    self.alpha_module = TemperatureCoef(alpha_init)
                    alpha_lr = float(self.alpha_cfg.get("lr", 1.0e-3))
                elif alpha_type == "const":
                    self.alpha_module = ConstantTemperatureCoef(alpha_init)
                    alpha_lr = 0.0
                else:
                    raise NotImplementedError(f"Alpha type {alpha_type} not supported")

            self.alpha_state = TrainState.create(
                apply_fn=self.alpha_module.apply,
                params=self.alpha_module.init({"params": alpha_key}, 0.0)["params"],
                tx=optax.adam(learning_rate=alpha_lr),
            )

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in cfg_get(self.cfg, "alg.critic.hs", default=[])]
        if isinstance(self.alpha_cfg, (int, float)):
            alpha_type = "const"
            alpha_init = float(self.alpha_cfg)
            alpha_lr = 0.0
        else:
            alpha_type = str(self.alpha_cfg["type"])
            alpha_init = float(self.alpha_cfg["init"])
            alpha_lr = float(self.alpha_cfg.get("lr", 0.0))
        return {
            "family": "soft_gac",
            "actor_base": self.actor_base_distribution,
            "actor_hidden_size": int(self.cfg.alg.actor.hidden_size),
            "actor_num_layers": int(self.cfg.alg.actor.num_layers),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "critic_crossq": True,
            "critic_update": "crossq_c51",
            "bridge_steps": int(self.bridge_steps),
            "policy_delay": int(self.policy_delay),
            "num_particles": int(self.num_particles),
            "particle_scheme": str(self.particle_scheme),
            "mix_recent": bool(self.mix_recent_replay),
            "recent_ratio": float(self.recent_replay_ratio),
            "recent_window": int(self.recent_replay_window),
            "bn": bool(self.critic_use_batch_norm),
            "layer_norm": bool(self.critic_use_layer_norm),
            "optimizer_bn": bool(self.cfg.alg.optimizer.bn),
            "bn_mode": str(self.cfg.alg.optimizer.bn_mode),
            "bn_momentum": float(self.cfg.alg.optimizer.bn_momentum),
            "bn_warmup": int(self.cfg.alg.optimizer.bn_warmup),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "alpha_type": alpha_type,
            "alpha_init": alpha_init,
            "alpha_lr": alpha_lr,
            "target_control_energy": float(self.target_control_energy),
            "actor_params": self._count_params(self.policy.actor_state.params),
            "critic_params": self._count_params(self.policy.qf_state.params),
        }

    @staticmethod
    def _path_control_cost(latents, drifts, sigmas):
        h = jnp.asarray(1.0 / drifts.shape[1], dtype=latents.dtype)
        sigma_sq = jnp.square(sigmas) + 1e-6
        ref_shift = -2.0 * h * jnp.tanh(latents[:, :-1, :])
        actor_shift = h * drifts
        term = sigma_sq + jnp.square(actor_shift - ref_shift) / (2.0 * h) - 1.0 - jnp.log(sigma_sq)
        return 0.5 * jnp.sum(term, axis=(-1, -2))

    @staticmethod
    def _sample_bridge_noise(
        key,
        batch_size: int,
        actor_noise_steps: int,
        action_dim: int,
        base_distribution: str,
    ):
        base_key, step_key = jax.random.split(key)
        if base_distribution == "logistic":
            eps = jnp.asarray(1e-6, dtype=jnp.float32)
            base_action = jax.random.uniform(
                base_key,
                (batch_size, action_dim),
                minval=-1.0 + eps,
                maxval=1.0 - eps,
                dtype=jnp.float32,
            )
            base_latent = jnp.arctanh(base_action)
        elif base_distribution == "normal":
            base_latent = jax.random.normal(
                base_key,
                (batch_size, action_dim),
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unsupported SoftGAC base_distribution: {base_distribution}")
        step_noise = jax.random.normal(
            step_key,
            (batch_size, actor_noise_steps - 1, action_dim),
            dtype=jnp.float32,
        )
        return jnp.concatenate([base_latent[:, None, :], step_noise], axis=1)

    @staticmethod
    def _sample_bridge_noise_components(
        key,
        batch_size: int,
        num_particles: int,
        actor_noise_steps: int,
        action_dim: int,
        base_distribution: str,
    ):
        base_key, step_key = jax.random.split(key)
        if base_distribution == "logistic":
            eps = jnp.asarray(1e-6, dtype=jnp.float32)
            base_action = jax.random.uniform(
                base_key,
                (batch_size, num_particles, action_dim),
                minval=-1.0 + eps,
                maxval=1.0 - eps,
                dtype=jnp.float32,
            )
            base_latent = jnp.arctanh(base_action)
        elif base_distribution == "normal":
            base_latent = jax.random.normal(
                base_key,
                (batch_size, num_particles, action_dim),
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unsupported SoftGAC base_distribution: {base_distribution}")
        step_noise = jax.random.normal(
            step_key,
            (batch_size, num_particles, actor_noise_steps - 1, action_dim),
            dtype=jnp.float32,
        )
        return base_latent, step_noise

    @staticmethod
    def _pack_particle_noises(base_latent, step_noise, actor_noise_steps: int, action_dim: int):
        batch_size, num_particles = base_latent.shape[:2]
        noises = jnp.concatenate([base_latent[:, :, None, :], step_noise], axis=2)
        return noises.reshape((batch_size * num_particles, actor_noise_steps, action_dim))

    @staticmethod
    def _sample_actor_particle_noises(
        key,
        batch_size: int,
        num_particles: int,
        actor_noise_steps: int,
        action_dim: int,
        base_distribution: str,
        particle_scheme: str,
    ):
        if particle_scheme == "random" or num_particles <= 1:
            return SoftGAC._sample_bridge_noise(
                key,
                batch_size * num_particles,
                actor_noise_steps,
                action_dim,
                base_distribution,
            )
        if particle_scheme != "antithetic":
            raise ValueError(f"Unsupported SoftGAC particle_scheme: {particle_scheme}")

        pair_count = num_particles // 2
        odd_count = num_particles % 2
        pair_key, odd_key = jax.random.split(key)
        base_half, step_half = SoftGAC._sample_bridge_noise_components(
            pair_key,
            batch_size,
            pair_count,
            actor_noise_steps,
            action_dim,
            base_distribution,
        )
        base_particles = jnp.concatenate([base_half, -base_half], axis=1)
        step_particles = jnp.concatenate([step_half, -step_half], axis=1)

        if odd_count:
            odd_base, odd_step = SoftGAC._sample_bridge_noise_components(
                odd_key,
                batch_size,
                odd_count,
                actor_noise_steps,
                action_dim,
                base_distribution,
            )
            base_particles = jnp.concatenate([base_particles, odd_base], axis=1)
            step_particles = jnp.concatenate([step_particles, odd_step], axis=1)

        return SoftGAC._pack_particle_noises(base_particles, step_particles, actor_noise_steps, action_dim)

    @staticmethod
    def _soft_q_from_dist(q_dist, z_atoms):
        q1, q2, min_q = twin_q_expectation(q_dist, z_atoms)
        return min_q, q1, q2

    @staticmethod
    @jax.jit
    def update_alpha(target_control_energy: float, alpha_state: TrainState, control_cost: float):
        def alpha_loss_fn(alpha_params):
            alpha_value = alpha_state.apply_fn({"params": alpha_params}, 0)
            return alpha_value * (target_control_energy - jax.lax.stop_gradient(control_cost))

        alpha_loss, grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
        alpha_state = alpha_state.apply_gradients(grads=grads)
        alpha_value = alpha_state.apply_fn({"params": alpha_state.params}, 0)
        return alpha_state, alpha_loss, alpha_value

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=["path_apply", "num_atoms", "actor_noise_steps", "action_dim", "base_distribution"],
    )
    def update_critic(
        gamma: float,
        target_actor_state: TrainState,
        qf_state: RLTrainState,
        next_observations: np.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        num_atoms: int,
        z_atoms,
        v_min: float,
        v_max: float,
        dist_entropy_coeff: float,
        alpha: float,
        action_scale,
        action_bias,
        actor_noise_steps: int,
        action_dim: int,
        base_distribution: str,
        key,
        path_apply,
    ):
        batch_size = rewards.shape[0]
        key, noise_key = jax.random.split(key)
        next_noises = SoftGAC._sample_bridge_noise(
            noise_key,
            batch_size,
            actor_noise_steps,
            action_dim,
            base_distribution,
        )
        next_final_z, next_latents, next_drifts, next_sigmas = path_apply(
            {"params": target_actor_state.params},
            next_observations,
            next_noises,
        )
        next_state_actions = jax.lax.stop_gradient(jnp.tanh(next_final_z) * action_scale + action_bias)
        next_control_cost = jax.lax.stop_gradient(
            SoftGAC._path_control_cost(next_latents, next_drifts, next_sigmas)
        )
        qf_state, metrics, key, _ = update_crossq_c51_critic(
            gamma,
            qf_state,
            observations,
            actions,
            next_observations,
            next_state_actions,
            rewards,
            dones,
            alpha * next_control_cost,
            num_atoms,
            z_atoms,
            v_min,
            v_max,
            dist_entropy_coeff,
            key,
        )
        metrics = dict(metrics, next_control_cost=next_control_cost.mean())
        return qf_state, metrics, key

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "num_particles",
            "actor_noise_steps",
            "action_dim",
            "base_distribution",
            "particle_scheme",
            "path_apply",
        ],
    )
    def _update_actor_direct(
        actor_state: TrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        alpha: float,
        z_atoms,
        action_scale,
        action_bias,
        num_particles: int,
        actor_noise_steps: int,
        action_dim: int,
        base_distribution: str,
        particle_scheme: str,
        key,
        path_apply,
    ):
        batch_size = observations.shape[0]
        key, noise_key, dropout_key = jax.random.split(key, 3)
        noises = SoftGAC._sample_actor_particle_noises(
            noise_key,
            batch_size,
            num_particles,
            actor_noise_steps,
            action_dim,
            base_distribution,
            particle_scheme,
        )

        def loss_fn(params):
            states_exp = jnp.repeat(observations, num_particles, axis=0)
            final_z, latents, drifts, sigmas = path_apply({"params": params}, states_exp, noises)
            actions = jnp.tanh(final_z) * action_scale + action_bias
            q_dist = qf_state.apply_fn(
                {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
                states_exp,
                actions,
                rngs={"dropout": dropout_key},
                train=False,
            )
            q_soft, _, _ = SoftGAC._soft_q_from_dist(q_dist, z_atoms)
            control_cost = SoftGAC._path_control_cost(latents, drifts, sigmas)
            actor_loss = (alpha * control_cost - q_soft).mean()
            return actor_loss, (control_cost.mean(), q_soft.mean())

        (actor_loss_value, (control_cost_mean, q_soft_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, actor_loss_value, control_cost_mean, q_soft_mean, key

    @staticmethod
    @jax.jit
    def _soft_update_target_actor(policy_tau: float, actor_state: TrainState, target_actor_state: TrainState):
        return target_actor_state.replace(
            params=optax.incremental_update(actor_state.params, target_actor_state.params, policy_tau)
        )

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "gradient_steps",
            "policy_delay_indices",
            "num_particles",
            "action_dim",
            "actor_noise_steps",
            "base_distribution",
            "particle_scheme",
            "num_atoms",
            "path_apply",
        ],
    )
    def _train(
        cls,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        qf_state: RLTrainState,
        actor_state: TrainState,
        target_actor_state: TrainState,
        alpha_state: TrainState,
        key,
        gamma: float,
        policy_tau: float,
        num_particles: int,
        action_dim: int,
        actor_noise_steps: int,
        base_distribution: str,
        particle_scheme: str,
        action_scale,
        action_bias,
        num_atoms: int,
        z_atoms,
        v_min: float,
        v_max: float,
        dist_entropy_coeff: float,
        target_control_energy: float,
        path_apply,
    ):
        actor_loss_value = jnp.array(0.0)
        alpha_loss_value = jnp.array(0.0)
        log_metrics = {}

        for i in range(gradient_steps):
            def slice_batch(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                step_batch_size = x.shape[0] // gradient_steps
                return x[step_batch_size * step: step_batch_size * (step + 1)]

            batch = ReplayBufferSamplesNp(
                observations=slice_batch(data.observations),
                actions=slice_batch(data.actions),
                next_observations=slice_batch(data.next_observations),
                dones=slice_batch(data.dones),
                rewards=slice_batch(data.rewards),
            )

            alpha_value = alpha_state.apply_fn({"params": alpha_state.params}, 0)

            qf_state, critic_metrics, key = cls.update_critic(
                gamma,
                target_actor_state,
                qf_state,
                batch.next_observations,
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.dones,
                num_atoms,
                z_atoms,
                v_min,
                v_max,
                dist_entropy_coeff,
                alpha_value,
                action_scale,
                action_bias,
                actor_noise_steps,
                action_dim,
                base_distribution,
                key,
                path_apply,
            )
            log_metrics = dict(log_metrics, **critic_metrics)

            if i in policy_delay_indices:
                actor_state, actor_loss_value, actor_control_cost, actor_q_soft, key = cls._update_actor_direct(
                    actor_state,
                    qf_state,
                    batch.observations,
                    alpha_value,
                    z_atoms,
                    action_scale,
                    action_bias,
                    num_particles,
                    actor_noise_steps,
                    action_dim,
                    base_distribution,
                    particle_scheme,
                    key,
                    path_apply,
                )
                alpha_state, alpha_loss_value, alpha_value = cls.update_alpha(
                    target_control_energy,
                    alpha_state,
                    actor_control_cost,
                )
                target_actor_state = cls._soft_update_target_actor(
                    policy_tau,
                    actor_state,
                    target_actor_state,
                )
                log_metrics["actor_loss"] = actor_loss_value
                log_metrics["actor_control_cost"] = actor_control_cost
                log_metrics["actor_soft_q"] = actor_q_soft
                log_metrics["alpha_loss"] = alpha_loss_value
                log_metrics["alpha"] = alpha_value

            log_metrics["target_control_energy"] = jnp.asarray(target_control_energy)

        return qf_state, actor_state, target_actor_state, alpha_state, key, log_metrics

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

        policy_delay_indices = {
            i: True for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0
        }
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)
        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        replay_data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )
        z_atoms = jnp.linspace(self.cfg.alg.critic.v_min, self.cfg.alg.critic.v_max, self.cfg.alg.critic.n_atoms)

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.alpha_state,
            self.key,
            log_metrics,
        ) = self._train(
            gradient_steps,
            replay_data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.alpha_state,
            self.key,
            cfg_get(self.cfg, "alg.train.gamma", "alg.gamma"),
            self.policy_tau,
            self.num_particles,
            self.action_dim,
            self.actor_noise_steps,
            self.actor_base_distribution,
            self.particle_scheme,
            self.policy.action_scale,
            self.policy.action_bias,
            self.cfg.alg.critic.n_atoms,
            z_atoms,
            self.cfg.alg.critic.v_min,
            self.cfg.alg.critic.v_max,
            self.cfg.alg.critic.dist_entropy_coeff,
            self.target_control_energy,
            self.policy.path_apply,
        )

        self._n_updates += gradient_steps

        metric_prefix = f"train/{self.cfg.alg.name}"
        self.logger.record(f"{metric_prefix}/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in log_metrics.items():
            try:
                log_val = v.item()
            except Exception:
                log_val = v
            self.logger.record(f"{metric_prefix}/{k}", log_val)
