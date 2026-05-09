from functools import partial

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from models.actor import FlowPolicy
from models.critic import VectorCritic
from models.utils import activation_fn


@flax.struct.dataclass
class FLACState:
    policy_params: dict
    qf_state: RLTrainState
    policy_opt_state: dict
    log_alpha: jnp.ndarray
    alpha_opt_state: dict
    critic_updates: int
    policy_updates: int


class FLACPolicy(BaseJaxPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box, cfg, squash_output: bool = True, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            features_extractor_kwargs=None,
            squash_output=squash_output,
        )
        self.cfg = cfg
        self.use_sde = False
        self._actor_iter_steps = int(cfg.alg.actor.iter_steps)
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
        act = jnp.array([self.action_space.sample()])
        t = jnp.zeros((1, 1))
        d_a = int(np.prod(self.action_space.shape))

        self.critic_use_batch_norm = bool(self.cfg.alg.optimizer.bn)
        self.critic_use_layer_norm = bool(self.cfg.alg.critic.use_layer_norm)
        self.qf = VectorCritic(
            dropout_rate=self.cfg.alg.critic.dropout_rate,
            use_layer_norm=self.critic_use_layer_norm,
            use_batch_norm=self.critic_use_batch_norm,
            bn_warmup=self.cfg.alg.optimizer.bn_warmup,
            batch_norm_momentum=self.cfg.alg.optimizer.bn_momentum,
            batch_norm_mode=self.cfg.alg.optimizer.bn_mode,
            net_arch=self.cfg.alg.critic.hs,
            activation_fn=activation_fn[self.cfg.alg.critic.activation],
            n_critics=self.cfg.alg.critic.n_critics,
            n_atoms=self.cfg.alg.critic.n_atoms,
        )
        self.policy_module = FlowPolicy(
            hidden_dim=int(self.cfg.alg.actor.hidden_size),
            num_actions=d_a,
            steps=self._actor_iter_steps,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
            num_layers=int(self.cfg.alg.actor.num_layers),
        )

        critic_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            act,
            train=False,
        )
        critic_batch_stats = critic_variables["batch_stats"]
        policy_variables = self.policy_module.init(actor_key, obs, act, t, train=True)
        init_log_alpha = jnp.array([float(self.cfg.alg.flac.init_log_alpha)])

        policy_tx = optax.adam(float(self.cfg.alg.optimizer.lr_actor))
        critic_tx = optax.adam(
            learning_rate=float(self.cfg.alg.optimizer.lr_critic),
            b1=float(self.cfg.alg.optimizer.b1),
            b2=0.999,
        )
        alpha_tx = optax.adam(float(self.cfg.alg.optimizer.lr_alpha))

        self.state = FLACState(
            policy_params=policy_variables["params"],
            qf_state=RLTrainState.create(
                apply_fn=self.qf.apply,
                params=critic_variables["params"],
                batch_stats=critic_batch_stats,
                target_params=critic_variables["params"],
                target_batch_stats=critic_batch_stats,
                tx=critic_tx,
            ),
            policy_opt_state=policy_tx.init(policy_variables["params"]),
            log_alpha=init_log_alpha,
            alpha_opt_state=alpha_tx.init(init_log_alpha),
            critic_updates=0,
            policy_updates=0,
        )
        self.policy_optimizer = policy_tx
        self.critic_module = self.qf
        self.alpha_optimizer = alpha_tx
        self.sampler = partial(self.policy_module.apply, method=self.policy_module.sample)
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler"])
    def sample_action(policy_params, observations, key, sampler):
        return sampler({"params": policy_params}, observations, rng=key, train=False)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del deterministic
        if not self.use_sde:
            self.reset_noise()
        actions = FLACPolicy.sample_action(self.state.policy_params, observation, self.noise_key, self.sampler)
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)
