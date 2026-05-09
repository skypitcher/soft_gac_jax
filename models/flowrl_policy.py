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
from models.critic import FlowRLBufferCritic, FlowRLValueNet, VectorCritic
from models.utils import activation_fn


@flax.struct.dataclass
class FlowRLState:
    policy_params: dict
    qf_state: RLTrainState
    critic_buf_params: dict
    critic_target_buf_params: dict
    v_buf_params: dict
    policy_opt_state: dict
    critic_buf_opt_state: dict
    v_buf_opt_state: dict
    critic_updates: int
    policy_updates: int


class FlowRLPolicy(BaseJaxPolicy):
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
        key, actor_key, qf_key, dropout_key, bn_key, buf_key, v_key = jax.random.split(key, 7)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        act = jnp.array([self.action_space.sample()])
        t = jnp.zeros((1, 1))
        d_a = int(np.prod(self.action_space.shape))

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
        self.critic_buf_module = FlowRLBufferCritic(
            hidden_dim=int(self.cfg.alg.flowrl.buffer_hidden_size),
            critic_num_layers=int(self.cfg.alg.flowrl.buffer_num_layers),
            activation="gelu",
        )
        self.v_buf_module = FlowRLValueNet(
            hidden_dim=int(self.cfg.alg.flowrl.buffer_hidden_size),
            output_dim=1,
            num_layers=int(self.cfg.alg.flowrl.buffer_num_layers),
            activation="gelu",
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
        critic_buf_variables = self.critic_buf_module.init(buf_key, obs, act)
        v_buf_variables = self.v_buf_module.init(v_key, obs)

        policy_tx = optax.adam(float(self.cfg.alg.optimizer.lr_actor))
        critic_tx = optax.adam(
            learning_rate=float(self.cfg.alg.optimizer.lr_critic),
            b1=float(self.cfg.alg.optimizer.b1),
            b2=0.999,
        )
        critic_buf_tx = optax.adam(float(self.cfg.alg.optimizer.lr_critic))
        v_buf_tx = optax.adam(float(self.cfg.alg.optimizer.lr_critic))

        self.state = FlowRLState(
            policy_params=policy_variables["params"],
            qf_state=RLTrainState.create(
                apply_fn=self.qf.apply,
                params=critic_variables["params"],
                batch_stats=critic_batch_stats,
                target_params=critic_variables["params"],
                target_batch_stats=critic_batch_stats,
                tx=critic_tx,
            ),
            critic_buf_params=critic_buf_variables["params"],
            critic_target_buf_params=jax.tree.map(jnp.copy, critic_buf_variables["params"]),
            v_buf_params=v_buf_variables["params"],
            policy_opt_state=policy_tx.init(policy_variables["params"]),
            critic_buf_opt_state=critic_buf_tx.init(critic_buf_variables["params"]),
            v_buf_opt_state=v_buf_tx.init(v_buf_variables["params"]),
            critic_updates=0,
            policy_updates=0,
        )
        self.policy_optimizer = policy_tx
        self.critic_module = self.qf
        self.critic_buf_optimizer = critic_buf_tx
        self.v_buf_optimizer = v_buf_tx
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
        actions = FlowRLPolicy.sample_action(self.state.policy_params, observation, self.noise_key, self.sampler)
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)
