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
from models.actor import DenoisingNetwork
from models.critic import VectorCritic
from models.legacy_utils import cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule
from models.utils import activation_fn


@flax.struct.dataclass
class QSMState:
    qf_state: RLTrainState
    actor_params: dict
    actor_opt_state: dict
    critic_updates: int
    actor_updates: int


class QSMPolicy(BaseJaxPolicy):
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
        self._T = int(cfg.alg.actor.iter_steps)
        self._time_embed_dim = int(cfg.alg.actor.time_embed_dim)
        self._ddpm_temperature = float(cfg.alg.qsm.ddpm_temperature)
        self._clip_sampler = bool(cfg.alg.qsm.clip_sampler)
        self._beta_schedule = str(cfg.alg.qsm.beta_schedule).lower()
        self._d_a = int(np.prod(action_space.shape))
        self.action_high = jnp.array(action_space.high, dtype=jnp.float32)
        self.action_low = jnp.array(action_space.low, dtype=jnp.float32)

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        act = jnp.array([self.action_space.sample()])
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)

        if self._beta_schedule == "vp":
            betas, alphas_cumprod = vp_beta_schedule(self._T)
        elif self._beta_schedule == "cosine":
            betas, alphas_cumprod = cosine_beta_schedule(self._T)
        elif self._beta_schedule == "linear":
            betas, alphas_cumprod = linear_beta_schedule(self._T)
        else:
            raise ValueError(f"Invalid beta_schedule='{self._beta_schedule}'")
        self._sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self._sqrt_recip_alphas = 1.0 / jnp.sqrt(1.0 - betas)
        self._noise_coeff = betas / jnp.sqrt(1.0 - alphas_cumprod)
        self._sqrt_betas = jnp.sqrt(betas)

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
        self.denoiser_module = DenoisingNetwork(
            hidden_dim=int(self.cfg.alg.actor.hidden_size),
            action_dim=self._d_a,
            time_embed_dim=self._time_embed_dim,
            num_layers=int(self.cfg.alg.actor.num_layers),
        )

        critic_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            act,
            train=False,
        )
        critic_batch_stats = critic_variables["batch_stats"]
        actor_params = self.denoiser_module.init(actor_key, act, dummy_t, obs)

        critic_tx = optax.adam(
            learning_rate=float(self.cfg.alg.optimizer.lr_critic),
            b1=float(self.cfg.alg.optimizer.b1),
            b2=0.999,
        )
        actor_tx = optax.adam(float(self.cfg.alg.optimizer.lr_actor))
        self.state = QSMState(
            qf_state=RLTrainState.create(
                apply_fn=self.qf.apply,
                params=critic_variables["params"],
                batch_stats=critic_batch_stats,
                target_params=critic_variables["params"],
                target_batch_stats=critic_batch_stats,
                tx=critic_tx,
            ),
            actor_params=actor_params,
            actor_opt_state=actor_tx.init(actor_params),
            critic_updates=0,
            actor_updates=0,
        )
        self.critic_module = self.qf
        self.actor_optimizer = actor_tx
        return key

    @partial(jax.jit, static_argnums=(0,))
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

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del deterministic
        if not self.use_sde:
            self.reset_noise()
        actions = self._reverse_sample(self.state.actor_params, observation, self.noise_key)
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)
