"""DIMDIME: Diffusion-Based Maximum Entropy Reinforcement LearningE

paper: https://arxiv.org/pdf/2502.02316
code: https://github.com/ALRhub/DIME
"""

import os
import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from gymnasium import spaces
from functools import partial

from models.dime_policy import DiffPol
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from common.off_policy_algorithm import OffPolicyAlgorithmJax
from common.crossq_c51 import twin_q_expectation, update_crossq_c51_critic
from common.label_utils import replay_mix_recent_label
from common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from common.buffers import sample_replay_per_step


def _sanitize_label_part(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _env_parts(cfg) -> tuple[str, str]:
    env_name = str(cfg.env_name)
    if "/" in env_name:
        domain, task = env_name.split("/", 1)
    else:
        domain, task = "", env_name
    return _sanitize_label_part(domain), _sanitize_label_part(task)


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return jax.lax.stop_gradient(self.ent_coef_init)


class DIME(OffPolicyAlgorithmJax):
    TARGET_ENTROPY_PER_ACTION_DIM: ClassVar[float] = 4.0

    @staticmethod
    def default_run_label(cfg) -> str:
        domain, task = _env_parts(cfg)
        env_label = f"{domain}_{task}" if domain else task
        diff_steps = int(cfg.alg.actor.diff_steps)
        policy_delay = int(cfg.alg.train.policy_delay)
        score_hidden = int(cfg.sampler.score_model.num_hid)
        score_layers = int(cfg.sampler.score_model.num_layers)

        critic_hs = [int(h) for h in cfg.alg.critic.hs]
        critic_layers = len(critic_hs)
        if len(set(critic_hs)) == 1:
            critic_hidden = str(critic_hs[0])
        else:
            critic_hidden = "-".join(str(h) for h in critic_hs)

        return (
            f"dime_{env_label}_"
            f"nfe{diff_steps}_"
            f"pd{policy_delay}_"
            f"s{score_hidden}x{score_layers}_"
            f"c{critic_hidden}x{critic_layers}_"
            f"{replay_mix_recent_label(cfg)}"
        )

    policy_aliases: ClassVar[Dict[str, Type[DiffPol]]] = {  # type: ignore[assignment]
        "MlpPolicy": DiffPol,
        # Minimal dict support using flatten()
        "MultiInputPolicy": DiffPol,
    }

    policy: DiffPol
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(self,
                 policy,
                 env: Union[GymEnv, str],
                 model_save_path: str,
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
            learning_rate=cfg.alg.optimizer.lr_actor,
            qf_learning_rate=cfg.alg.optimizer.lr_critic,
            buffer_size=cfg.alg.replay.buffer_size,
            learning_starts=cfg.alg.train.learning_starts,
            batch_size=cfg.alg.train.batch_size,
            tau=cfg.alg.target.critic_tau,
            gamma=cfg.alg.train.gamma,
            train_freq=train_freq,
            gradient_steps=cfg.alg.train.utd,
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
        self.policy_delay = self.cfg.alg.train.policy_delay
        self.ent_coef_params = self.cfg.alg.ent_coef
        self.policy_q_reduce_fn = jax.numpy.mean
        self.save_every_n_steps = save_every_n_steps
        self.model_save_path = model_save_path
        self.policy_tau = self.cfg.alg.target.actor_tau
        self.mix_recent_replay = bool(self.cfg.alg.replay.get("mix_recent", True))
        self.recent_replay_ratio = float(self.cfg.alg.replay.get("recent_ratio", 0.2))
        self.recent_replay_window = int(self.cfg.alg.replay.get("recent_window", 4096))
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self, reset=False) -> None:
        if not reset:
            super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None or reset:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.cfg
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if self.ent_coef_params["type"] == "auto":
                    ent_coef_init = self.ent_coef_params['init']
                    # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                    # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                    self.ent_coef = EntropyCoef(ent_coef_init)
            elif self.ent_coef_params["type"] == "const":
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_params["init"], float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_params['init']}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_params["init"])  # type: ignore[assignment]
            else:
                raise NotImplementedError(f"Entropy coefficient type {self.ent_coef_params['type']} not supported")

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init({"params": ent_key}, 0.0)["params"],
                tx=optax.adam(
                    # learning_rate=self.learning_rate,
                    learning_rate=1.0e-3,
                ),
            )

            # DIME tunes its coefficient toward the path run-cost budget used by
            # the official implementation, not endpoint action entropy.
            self.target_entropy = self.action_space.shape[0] * self.TARGET_ENTROPY_PER_ACTION_DIM

    @staticmethod
    def _count_params(params) -> int:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def describe_model(self) -> dict[str, Any]:
        critic_hs = [int(h) for h in self.cfg.alg.critic.hs]
        return {
            "family": "dime",
            "diff_steps": int(self.cfg.alg.actor.diff_steps),
            "score_hidden_size": int(self.cfg.sampler.score_model.num_hid),
            "score_num_layers": int(self.cfg.sampler.score_model.num_layers),
            "critic_hidden": critic_hs,
            "critic_num_layers": len(critic_hs),
            "policy_delay": int(self.policy_delay),
            "bn": bool(self.cfg.alg.optimizer.bn),
            "bn_mode": str(self.cfg.alg.optimizer.bn_mode),
            "bn_momentum": float(self.cfg.alg.optimizer.bn_momentum),
            "bn_warmup": int(self.cfg.alg.optimizer.bn_warmup),
            "lr_actor": float(self.cfg.alg.optimizer.lr_actor),
            "lr_critic": float(self.cfg.alg.optimizer.lr_critic),
            "actor_tau": float(self.policy_tau),
            "mix_recent": bool(self.mix_recent_replay),
            "recent_ratio": float(self.recent_replay_ratio),
            "recent_window": int(self.recent_replay_window),
            "ent_coef_type": str(self.ent_coef_params["type"]),
            "ent_coef_init": float(self.ent_coef_params["init"]),
            "target_run_cost": float(self.target_entropy),
            "target_run_cost_per_action_dim": float(self.TARGET_ENTROPY_PER_ACTION_DIM),
            "actor_params": self._count_params(self.policy.actor_state.params),
            "critic_params": self._count_params(self.policy.qf_state.params),
        }

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
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if
                                ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.gamma,
            self.policy_tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            self.num_timesteps,
            self.policy_q_reduce_fn,
            self.policy.sampler,
            self.policy.target_sampler,
            self.cfg.alg.critic.v_min,
            self.cfg.alg.critic.v_max,
            self.cfg.alg.critic.dist_entropy_coeff,
            self.cfg.alg.critic.n_atoms
        )
        self._n_updates += gradient_steps

        if self.model_save_path is not None:
            if (self.num_timesteps % self.save_every_n_steps == 0) or (self.num_timesteps == (self.learning_starts+1)):
                self._save_model()

        metric_prefix = f"train/{self.cfg.alg.name}"
        self.logger.record(f"{metric_prefix}/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in log_metrics.items():
            try:
                log_val = v.item()
            except:
                log_val = v
            self.logger.record(f"{metric_prefix}/{k}", log_val)

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "num_atoms", "v_min", "v_max", "dist_entropy_coeff"])
    def update_critic(
            gamma: float,
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            actions: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            n_env_interacts: int,
            num_atoms: int,
            z_atoms: jnp.ndarray,
            v_min: int,
            v_max: int,
            dist_entropy_coeff: float,
            key,
            sampler
    ):
        key, noise_key = jax.random.split(key)
        # sample action from the actor

        out = DiffPol.sample_action(actor_state, actor_state.params, next_observations, noise_key, sampler)
        all_actions, next_run_costs, next_sto_costs, next_terminal_costs, latents, v_t = out
        next_state_actions = jax.lax.stop_gradient(all_actions)
        next_run_costs = jax.lax.stop_gradient(next_run_costs)
        next_sto_costs = jax.lax.stop_gradient(next_sto_costs)
        next_terminal_costs = jax.lax.stop_gradient(next_terminal_costs)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
        next_penalty = ent_coef_value * (next_run_costs + next_sto_costs + next_terminal_costs)
        qf_state, metrics, key, _ = update_crossq_c51_critic(
            gamma,
            qf_state,
            observations,
            actions,
            next_observations,
            next_state_actions,
            rewards,
            dones,
            next_penalty,
            num_atoms,
            z_atoms,
            v_min,
            v_max,
            dist_entropy_coeff,
            key,
        )
        metrics = dict(metrics, ent_coef=ent_coef_value)
        return qf_state, metrics, key

    @staticmethod
    @partial(jax.jit, static_argnames=["q_reduce_fn", "sampler"])
    def update_actor(
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            n_env_interacts: int,
            key,
            z_atoms: jnp.ndarray,
            sampler,
            q_reduce_fn,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(actor_state_in, actor_params):
            out = DiffPol.sample_action(actor_state_in, actor_params, observations, noise_key, sampler)
            actions, run_costs, sto_costs, terminal_costs, latents, v_t = out
            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats
                },
                observations,
                actions,
                rngs={"dropout": dropout_key}, train=False
            )

            qf_pi1, qf_pi2, _ = twin_q_expectation(qf_pi, z_atoms)
            min_qf_pi = q_reduce_fn(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
            actor_loss = (- min_qf_pi + ent_coef_value * (run_costs.squeeze() + sto_costs.squeeze() + terminal_costs.squeeze())).mean()

            max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
            min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
            mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

            latent_acts = {'max_la': max_actions, 'min_la': min_actions, 'mean_la': mean_actions}

            return actor_loss, (run_costs.mean(), sto_costs.mean(), terminal_costs.mean(), latent_acts)

        outs = jax.value_and_grad(actor_loss, has_aux=True, argnums=1)(actor_state, actor_state.params)
        (act_loss_value, (run_costs_mean, sto_costs, terminal_costs, latent_acts)), grads = outs
        actor_state = actor_state.apply_gradients(grads=grads)
        metrics = {"entropy": 0.0,
                   "run_costs": run_costs_mean,
                   "sto_costs": sto_costs,
                   "terminal_costs": terminal_costs,
        }
        return actor_state, qf_state, act_loss_value, key, [metrics, latent_acts]

    @staticmethod
    @jax.jit
    def soft_update_target_actor(tau: float, actor_state: TrainState, target_actor_state: TrainState):
        target_actor_state = target_actor_state.replace(
            params=optax.incremental_update(actor_state.params, target_actor_state.params, tau))
        return target_actor_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: np.ndarray, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params}, 0)
            ent_coef_loss = -ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @classmethod
    @partial(jax.jit,
             static_argnames=["cls", "gradient_steps", "q_reduce_fn",
                              "sampler", "target_sampler", "v_min", "v_max", "num_atoms", "dist_entropy_coeff"])
    def _train(
            cls,
            gamma: float,
            policy_tau: float,
            target_entropy: np.ndarray,
            gradient_steps: int,
            data: ReplayBufferSamplesNp,
            policy_delay_indices: flax.core.FrozenDict,
            qf_state: RLTrainState,
            actor_state: TrainState,
            target_actor_state: TrainState,
            ent_coef_state: TrainState,
            key,
            n_env_interacts,
            q_reduce_fn,
            sampler,
            target_sampler,
            v_min,
            v_max,
            dist_entropy_coeff,
            num_atoms
    ):
        actor_loss_value = jnp.array(0)
        actor_metrics = [{}]
        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step: batch_size * (step + 1)]

            z_atoms = jnp.linspace(v_min,  v_max, num_atoms)

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = cls.update_critic(
                gamma,
                target_actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                n_env_interacts,
                num_atoms,
                z_atoms,
                v_min,
                v_max,
                dist_entropy_coeff,
                key,
                target_sampler
            )
            target_actor_state = target_actor_state
            # hack to be able to jit (n_updates % policy_delay == 0)
            # a = False
            if i in policy_delay_indices:  # and a:
                (actor_state, qf_state, actor_loss_value, key, actor_metrics) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    n_env_interacts,
                    key,
                    z_atoms,
                    sampler,
                    q_reduce_fn,
                )
                ent_coef_state, _ = DIME.update_temperature(target_entropy, ent_coef_state,
                                                           actor_metrics[0]['run_costs'])

                target_actor_state = DIME.soft_update_target_actor(policy_tau, actor_state, target_actor_state)
        log_metrics = {'actor_loss': actor_loss_value, **actor_metrics[0], **log_metrics_critic}
        return qf_state, actor_state, target_actor_state, ent_coef_state, key, log_metrics

    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)

    def current_entropy_coeff(self):
        return self.ent_coef_state.apply_fn({"params": self.ent_coef_state.params})

    def _save_model(self):
        save_model_state(self.policy.actor_state, self.model_save_path, "actor_state", self.num_timesteps)
        save_model_state(self.policy.qf_state, self.model_save_path, "critic_state", self.num_timesteps)

    def load_model(self, path, n_steps_actor, n_steps_critic):
        self.policy.actor_state = load_state(path, "actor_state", n_steps_actor, train_state=self.policy.actor_state)
        self.policy.qf_state = load_state(path, "critic_state", n_steps_critic, train_state=self.policy.qf_state)


# Save and load model
def save_model_state(train_state, path, name, n_steps):
    # Serialize the model parameters
    serialized_state = flax.serialization.to_bytes(train_state)
    os.makedirs(path, exist_ok=True)
    extended_path = os.path.join(path, f'{name}_{n_steps}.msgpack')
    # Save the serialized parameters to a file
    with open(extended_path, 'wb') as f:
        f.write(serialized_state)


def load_state(path, name, n_steps, train_state=None):
    extended_path = os.path.join(path, f'{name}_{n_steps}.msgpack')
    # Load the serialized parameters from a file
    with open(extended_path, 'rb') as f:
        train_state_loaded = f.read()
    return flax.serialization.from_bytes(train_state, train_state_loaded)
