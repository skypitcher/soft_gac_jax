from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from common.config_utils import cfg_get
from common.policies import BaseJaxPolicy
from common.soft_gac_base import canonical_base_distribution
from common.type_aliases import RLTrainState
from models.actor import BridgePolicy
from models.critic import VectorCritic
from models.utils import activation_fn


class SoftGACPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        cfg,
        squash_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            features_extractor_kwargs=None,
            squash_output=squash_output,
        )
        self.cfg = cfg
        self.use_sde = False
        self.actor_noise_steps = int(cfg_get(cfg, "alg.actor.num_layers", "alg.actor_num_layers")) + 1
        self.actor_base_distribution = canonical_base_distribution(
            cfg_get(cfg, "alg.actor.base_distribution", default="logistic")
        )

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        action_scale = jnp.array((self.action_space.high - self.action_space.low) / 2.0, dtype=jnp.float32)
        action_bias = jnp.array((self.action_space.high + self.action_space.low) / 2.0, dtype=jnp.float32)

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
        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        qf_batch_stats = qf_init_variables["batch_stats"]
        target_qf_batch_stats = target_qf_init_variables["batch_stats"]
        qf_tx = optax.adam(
            learning_rate=qf_learning_rate,
            b1=self.cfg.alg.optimizer.b1,
            b2=0.999,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_batch_stats,
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_batch_stats,
            tx=qf_tx,
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=(
                "dropout_rate",
                "use_layer_norm",
                "use_batch_norm",
                "batch_norm_momentum",
                "batch_norm_mode",
            ),
        )

        self.actor = BridgePolicy(
            hidden_dim=int(cfg_get(self.cfg, "alg.actor.hidden_size", "alg.actor_hidden_size")),
            num_actions=self.action_space.shape[0],
            action_scale=action_scale,
            action_bias=action_bias,
            num_layers=int(cfg_get(self.cfg, "alg.actor.num_layers", "alg.actor_num_layers")),
            base_distribution=self.actor_base_distribution,
        )
        actor_noise = jnp.zeros((1, self.actor_noise_steps, self.action_space.shape[0]), dtype=jnp.float32)
        actor_params = self.actor.init(actor_key, obs, actor_noise)["params"]
        actor_tx = optax.adam(lr_schedule(1))
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.target_actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.set_to_zero(),
        )

        self.sampler = partial(self.actor.apply, method=self.actor.sample)
        self.path_apply = partial(self.actor.apply, method=self.actor.path_stats)
        self.action_scale = action_scale
        self.action_bias = action_bias
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler"])
    def sample_action(actor_state, actor_params, observations, key, sampler):
        return sampler({"params": actor_params}, observations, rng=key)

    @staticmethod
    @partial(jax.jit, static_argnames=["path_apply"])
    def path_stats(actor_state, actor_params, observations, noises, path_apply):
        return path_apply({"params": actor_params}, observations, noises)

    @staticmethod
    @partial(jax.jit, static_argnames=["actor_noise_steps"])
    def deterministic_action(actor_state, actor_params, observations, action_scale, action_bias, actor_noise_steps: int):
        noise = jnp.zeros((observations.shape[0], actor_noise_steps, action_scale.shape[0]), dtype=jnp.float32)
        z = actor_state.apply_fn({"params": actor_params}, observations, noise)
        return jnp.tanh(z) * action_scale + action_bias

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            actions = SoftGACPolicy.deterministic_action(
                self.actor_state,
                self.actor_state.params,
                observation,
                self.action_scale,
                self.action_bias,
                self.actor_noise_steps,
            )
        else:
            if not self.use_sde:
                self.reset_noise()
            actions = SoftGACPolicy.sample_action(
                self.actor_state,
                self.actor_state.params,
                observation,
                self.noise_key,
                self.sampler,
            )
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray):
        if not self.use_sde:
            self.reset_noise()

        def q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o,
                a,
                rngs={"dropout": dropout_key},
                train=False,
            )

        return jax.jit(q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )
