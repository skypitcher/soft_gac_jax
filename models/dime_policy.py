import jax
import numpy as np
import jax.numpy as jnp

from functools import partial

import optax
from gymnasium import spaces
from models.critic import VectorCritic
from algorithms.dime_impl.utils import get_sampler_init
from algorithms.dime_impl.od.od_integrators import get_integrator as get_integrator_od
from algorithms.dime_impl.od.od_sampling import sample as sample_od
from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from stable_baselines3.common.type_aliases import Schedule

from models.utils import activation_fn


class DiffPol(BaseJaxPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Box,
                 cfg,
                 squash_output: bool = True,
                 **kwargs,
                 ):
        super().__init__(observation_space,
                         action_space,
                         features_extractor=None,
                         features_extractor_kwargs=None,
                         squash_output=squash_output)
        self.cfg = cfg
        self.use_sde = False

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, score_key, stat_distr_key, qf_key, dropout_key, stat_distr_bn_key, bn_key = jax.random.split(key, 7)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        a_dim = self.action_space.shape[0]
        obs_dim = obs.shape[1]

        # initialize Q-function
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

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=optax.adam(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **dict({
                    'b1': self.cfg.alg.optimizer.b1,
                    'b2': 0.999  # default
                }),
            ),
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        # Initialize actor
        key, diff_key = jax.random.split(key, 2)
        self.actor_model, self.actor_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, obs_dim)
        target_model_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, obs_dim)
        self.actor_target_model, self.target_actor_state = target_model_state
        self.integrator = get_integrator_od(self.cfg, self.actor_model)
        self.target_integrator = get_integrator_od(self.cfg, self.actor_target_model)
        sampler = sample_od
        self.sampler = partial(sampler, integrator=self.integrator, diffusion_model=self.actor_model)
        self.target_sampler = partial(sampler, integrator=self.target_integrator,
                                      diffusion_model=self.actor_target_model)
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "return_logprob"])
    def sample_action(actor_state, actor_params, observations, key, sampler, return_logprob=False):
        out = sampler(key, actor_state, actor_params, observations, stop_grad=False)
        # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
        final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
        return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, *_ = DiffPol.sample_action(self.actor_state, self.actor_state.params, observation, self.noise_key,
                                            self.sampler)
        return actions[0]

    def _predict2(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, _, _, _, la, _ = DiffPol.sample_action(self.actor_state, self.actor_state.params, observation,
                                                        self.noise_key, self.sampler)
        actions = (actions, la)
        return actions

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a,
                rngs={"dropout": dropout_key},
                train=False
            )

        return jax.jit(Q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )
