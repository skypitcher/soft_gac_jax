from __future__ import annotations

from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from common.config_utils import cfg_get
from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from models.critic import BatchRenorm
from models.critic import VectorCritic
from models.utils import activation_fn


LOG_2PI = float(np.log(2.0 * np.pi))


@flax.struct.dataclass
class SACState:
    actor_params: dict
    actor_batch_stats: flax.core.FrozenDict
    qf_state: RLTrainState
    actor_opt_state: dict
    log_alpha: jnp.ndarray
    alpha_opt_state: dict
    critic_updates: int
    actor_updates: int


class SquashedGaussianActor(nn.Module):
    hidden_dim: int
    num_layers: int
    num_actions: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    batch_norm_mode: str = "brn_actor"
    bn_warmup: int = 100_000

    @nn.compact
    def __call__(self, obs, *, train: bool = False):
        if "brn_actor" in self.batch_norm_mode:
            norm_cls = BatchRenorm
        elif "bn" in self.batch_norm_mode or "brn" in self.batch_norm_mode:
            norm_cls = nn.BatchNorm
        else:
            norm_cls = nn.BatchNorm

        x = obs
        use_actor_norm = self.use_batch_norm and "noactor" not in self.batch_norm_mode
        def apply_norm(y, name):
            if norm_cls is BatchRenorm:
                return norm_cls(
                    bn_warmup=self.bn_warmup,
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    name=name,
                )(y)
            return norm_cls(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                name=name,
            )(y)

        if use_actor_norm:
            x = apply_norm(x, "input_norm")

        for i in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name=f"hidden_{i}",
            )(x)
            x = nn.relu(x)
            if use_actor_norm:
                x = apply_norm(x, f"hidden_{i}_norm")
        out = nn.Dense(
            2 * self.num_actions,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="out",
        )(x)
        mean, raw_log_std = jnp.split(out, 2, axis=-1)
        log_std = nn.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def _squashed_log_prob(self, pre_tanh, mean, log_std):
        std = jnp.exp(log_std)
        normal_log_prob = -0.5 * (jnp.square((pre_tanh - mean) / (std + 1e-8)) + 2.0 * log_std + LOG_2PI)
        normal_log_prob = normal_log_prob.sum(axis=-1, keepdims=True)
        squashed = jnp.tanh(pre_tanh)
        squash_correction = jnp.log(1.0 - jnp.square(squashed) + 1e-6).sum(axis=-1, keepdims=True)
        scale_correction = jnp.log(jnp.maximum(self.action_scale, 1e-6)).sum()
        return normal_log_prob - squash_correction - scale_correction

    def sample_and_log_prob(self, obs, *, rng, train: bool = False):
        mean, log_std = self(obs, train=train)
        noise = jax.random.normal(rng, mean.shape)
        pre_tanh = mean + jnp.exp(log_std) * noise
        action = jnp.tanh(pre_tanh) * self.action_scale + self.action_bias
        log_prob = self._squashed_log_prob(pre_tanh, mean, log_std)
        return action, log_prob

    def deterministic(self, obs, *, train: bool = False):
        mean, _ = self(obs, train=train)
        return jnp.tanh(mean) * self.action_scale + self.action_bias


class SACPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        cfg,
        squash_output: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            features_extractor_kwargs=None,
            squash_output=squash_output,
        )
        del kwargs
        self.cfg = cfg
        self.use_sde = False
        self.action_scale = jnp.array((action_space.high - action_space.low) / 2.0, dtype=jnp.float32)
        self.action_bias = jnp.array((action_space.high + action_space.low) / 2.0, dtype=jnp.float32)

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])
        action_dim = int(np.prod(self.action_space.shape))

        self.qf = VectorCritic(
            dropout_rate=self.cfg.alg.critic.dropout_rate,
            use_layer_norm=self.cfg.alg.critic.use_layer_norm,
            use_batch_norm=self.cfg.alg.optimizer.bn,
            bn_warmup=self.cfg.alg.optimizer.bn_warmup,
            batch_norm_momentum=self.cfg.alg.optimizer.bn_momentum,
            batch_norm_mode=self.cfg.alg.optimizer.bn_mode,
            net_arch=self.cfg.alg.critic.hs,
            activation_fn=activation_fn[self.cfg.alg.critic.activation],
            n_critics=self.cfg.alg.critic.n_critics,
            n_atoms=self.cfg.alg.critic.n_atoms,
        )
        qf_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        qf_batch_stats = qf_variables["batch_stats"]
        qf_tx = optax.adam(
            learning_rate=qf_learning_rate,
            b1=self.cfg.alg.optimizer.b1,
            b2=0.999,
        )

        self.actor_module = SquashedGaussianActor(
            hidden_dim=int(self.cfg.alg.actor.hidden_size),
            num_layers=int(self.cfg.alg.actor.num_layers),
            num_actions=action_dim,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            log_std_min=float(cfg_get(self.cfg, "alg.actor.log_std_min", default=-5.0)),
            log_std_max=float(cfg_get(self.cfg, "alg.actor.log_std_max", default=2.0)),
            use_batch_norm=bool(self.cfg.alg.optimizer.bn),
            batch_norm_momentum=float(self.cfg.alg.optimizer.bn_momentum),
            batch_norm_mode=str(self.cfg.alg.optimizer.bn_mode),
            bn_warmup=int(self.cfg.alg.optimizer.bn_warmup),
        )
        actor_variables = self.actor_module.init(
            {"params": actor_key, "batch_stats": bn_key},
            obs,
            train=False,
        )
        actor_batch_stats = actor_variables.get("batch_stats", flax.core.freeze({}))

        alpha_cfg = self.cfg.alg.alpha
        if isinstance(alpha_cfg, (int, float)):
            alpha_type = "const"
            alpha_init = float(alpha_cfg)
            alpha_lr = 0.0
        else:
            alpha_type = str(alpha_cfg.get("type", "auto"))
            alpha_init = float(alpha_cfg.init)
            alpha_lr = float(alpha_cfg.get("lr", 0.0)) if alpha_type == "auto" else 0.0
        if alpha_type == "const" and alpha_init <= 0.0:
            log_alpha = jnp.array([-jnp.inf], dtype=jnp.float32)
        else:
            log_alpha = jnp.array([jnp.log(jnp.maximum(alpha_init, 1e-8))], dtype=jnp.float32)
        actor_tx = optax.adam(float(lr_schedule(1)))
        alpha_tx = optax.adam(alpha_lr)

        self.state = SACState(
            actor_params=actor_variables["params"],
            actor_batch_stats=actor_batch_stats,
            qf_state=RLTrainState.create(
                apply_fn=self.qf.apply,
                params=qf_variables["params"],
                batch_stats=qf_batch_stats,
                target_params=qf_variables["params"],
                target_batch_stats=qf_batch_stats,
                tx=qf_tx,
            ),
            actor_opt_state=actor_tx.init(actor_variables["params"]),
            log_alpha=log_alpha,
            alpha_opt_state=alpha_tx.init(log_alpha),
            critic_updates=0,
            actor_updates=0,
        )
        self.actor_optimizer = actor_tx
        self.alpha_optimizer = alpha_tx
        self.sampler = partial(self.actor_module.apply, method=self.actor_module.sample_and_log_prob)
        self.deterministic_sampler = partial(self.actor_module.apply, method=self.actor_module.deterministic)
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler"])
    def sample_action(actor_params, actor_batch_stats, observations, key, sampler):
        action, _ = sampler({"params": actor_params, "batch_stats": actor_batch_stats}, observations, rng=key, train=False)
        return action

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler"])
    def deterministic_action(actor_params, actor_batch_stats, observations, sampler):
        return sampler({"params": actor_params, "batch_stats": actor_batch_stats}, observations, train=False)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            actions = SACPolicy.deterministic_action(
                self.state.actor_params,
                self.state.actor_batch_stats,
                observation,
                self.deterministic_sampler,
            )
        else:
            if not self.use_sde:
                self.reset_noise()
            actions = SACPolicy.sample_action(
                self.state.actor_params,
                self.state.actor_batch_stats,
                observation,
                self.noise_key,
                self.sampler,
            )
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)
