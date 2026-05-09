from functools import partial

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from common.crossq_c51 import twin_q_expectation
from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from models.actor import DiffusionDenoiser
from models.critic import VectorCritic
from models.legacy_utils import cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule
from models.utils import activation_fn


@flax.struct.dataclass
class QVPOState:
    actor_params: dict
    target_actor_params: dict
    qf_state: RLTrainState
    actor_opt_state: dict
    running_q_mean: jnp.ndarray
    running_q_std: jnp.ndarray
    critic_updates: int
    actor_updates: int


class QVPOPolicy(BaseJaxPolicy):
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
        self._T = int(cfg.alg.actor.n_timesteps)
        self._time_embed_dim = int(cfg.alg.actor.time_embed_dim)
        self._noise_ratio = float(cfg.alg.qvpo.noise_ratio)
        self._beta_schedule = str(cfg.alg.qvpo.beta_schedule).lower()
        self._clip_denoised = bool(cfg.alg.qvpo.clip_denoised)
        self._behavior_sample = int(cfg.alg.qvpo.behavior_sample)
        self._eval_sample = int(cfg.alg.qvpo.eval_sample)
        self._deterministic_sampling = bool(cfg.alg.qvpo.deterministic_sampling)
        self._d_a = int(np.prod(action_space.shape))
        self.action_high = jnp.ones((self._d_a,), dtype=jnp.float32)
        self.action_low = -jnp.ones((self._d_a,), dtype=jnp.float32)

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        del lr_schedule
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
        alphas = 1.0 - betas
        alphas_cumprod_prev = jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self._sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self._sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        self._sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)
        self._posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self._posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
        self._posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, min=1e-20))
        self._z_atoms = jnp.linspace(float(self.cfg.alg.critic.v_min), float(self.cfg.alg.critic.v_max), int(self.cfg.alg.critic.n_atoms))

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
        self.actor_module = DiffusionDenoiser(
            hidden_dim=int(self.cfg.alg.actor.hidden_size),
            action_dim=self._d_a,
            time_embed_dim=self._time_embed_dim,
            num_layers=int(self.cfg.alg.actor.num_layers),
            use_layer_norm=bool(self.cfg.alg.actor.use_layer_norm),
        )

        critic_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            act,
            train=False,
        )
        critic_batch_stats = critic_variables["batch_stats"]
        actor_variables = self.actor_module.init(actor_key, act, dummy_t, obs)

        actor_transforms = []
        ac_grad_norm = float(self.cfg.alg.qvpo.ac_grad_norm)
        if ac_grad_norm > 0:
            actor_transforms.append(optax.clip_by_global_norm(ac_grad_norm))
        actor_transforms.append(optax.adam(float(self.cfg.alg.optimizer.lr_actor), eps=1e-5))
        actor_tx = optax.chain(*actor_transforms)
        critic_tx = optax.adam(
            learning_rate=float(qf_learning_rate),
            b1=float(self.cfg.alg.optimizer.b1),
            b2=0.999,
        )

        self.state = QVPOState(
            actor_params=actor_variables["params"],
            target_actor_params=jax.tree.map(jnp.copy, actor_variables["params"]),
            qf_state=RLTrainState.create(
                apply_fn=self.qf.apply,
                params=critic_variables["params"],
                batch_stats=critic_batch_stats,
                target_params=critic_variables["params"],
                target_batch_stats=critic_batch_stats,
                tx=critic_tx,
            ),
            actor_opt_state=actor_tx.init(actor_variables["params"]),
            running_q_mean=jnp.array(0.0, dtype=jnp.float32),
            running_q_std=jnp.array(1.0, dtype=jnp.float32),
            critic_updates=0,
            actor_updates=0,
        )
        self.actor_optimizer = actor_tx
        self.critic_module = self.qf
        return key

    @partial(jax.jit, static_argnums=(0,))
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

    def _critic_eval(self, qf_state, obs, act):
        return qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            obs,
            act,
            train=False,
        )

    @partial(jax.jit, static_argnums=(0, 5))
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

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not self.use_sde:
            self.reset_noise()
        num_samples = self._eval_sample if deterministic else self._behavior_sample
        noise_ratio = 0.0 if deterministic and self._deterministic_sampling else self._noise_ratio
        actions = self._sample_best_actions(
            self.state.actor_params,
            self.state.qf_state,
            observation,
            self.noise_key,
            num_samples,
            noise_ratio,
        )
        return actions[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        del batch_size
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)
