import jax
import jax.numpy as jnp

from common.type_aliases import RLTrainState


def q_expectation(q_dist, z_atoms):
    return jnp.sum(q_dist * z_atoms, axis=-1)


def twin_q_expectation(q_dist, z_atoms):
    q1 = q_expectation(q_dist[0], z_atoms)
    q2 = q_expectation(q_dist[1], z_atoms)
    return q1, q2, jnp.minimum(q1, q2)


def _project_c51(next_dist, rewards, dones, gamma, v_min, v_max, num_atoms, support, next_penalty):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    target_z = jnp.clip(
        rewards[:, None] + (1.0 - dones[:, None]) * gamma * (support[None, :] - next_penalty[:, None]),
        v_min,
        v_max,
    )
    b = (target_z - v_min) / delta_z
    atom_idx = jnp.arange(num_atoms, dtype=next_dist.dtype)
    weights = jnp.maximum(0.0, 1.0 - jnp.abs(b[:, :, None] - atom_idx[None, None, :]))
    return jnp.sum(next_dist[:, :, None] * weights, axis=1)


def update_crossq_c51_critic(
    gamma: float,
    qf_state: RLTrainState,
    observations,
    actions,
    next_observations,
    next_actions,
    rewards,
    dones,
    next_penalty,
    num_atoms: int,
    z_atoms,
    v_min: float,
    v_max: float,
    dist_entropy_coeff: float,
    key,
):
    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)
    next_penalty = next_penalty.reshape(-1)
    batch_size = rewards.shape[0]
    key, dropout_key = jax.random.split(key)

    def ce_loss(params, batch_stats, dropout_key_):
        catted_q_values, state_updates = qf_state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            jnp.concatenate([observations, next_observations], axis=0),
            jnp.concatenate([actions, next_actions], axis=0),
            rngs={"dropout": dropout_key_},
            mutable=["batch_stats"],
            train=True,
        )
        current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
        current_q1 = current_q_values[0]
        current_q2 = current_q_values[1]
        next_q1 = next_q_values[0]
        next_q2 = next_q_values[1]

        target_q1 = _project_c51(next_q1, rewards, dones, gamma, v_min, v_max, num_atoms, z_atoms, next_penalty)
        target_q2 = _project_c51(next_q2, rewards, dones, gamma, v_min, v_max, num_atoms, z_atoms, next_penalty)
        target_q = jax.lax.stop_gradient(0.5 * (target_q1 + target_q2))

        def categorical_ce(pred, target):
            ce = -jnp.mean(jnp.sum(target * jnp.log(pred + 1e-15), axis=-1))
            entropy_reg = dist_entropy_coeff * jnp.mean(jnp.sum(pred * jnp.log(pred + 1e-15), axis=-1))
            return ce + entropy_reg

        loss = categorical_ce(current_q1, target_q) + categorical_ce(current_q2, target_q)
        _, _, current_q = twin_q_expectation(current_q_values, z_atoms)
        next_q = q_expectation(target_q, z_atoms)
        return loss, (state_updates, current_q, next_q)

    (critic_loss, (state_updates, current_q, next_q)), grads = jax.value_and_grad(ce_loss, has_aux=True)(
        qf_state.params,
        qf_state.batch_stats,
        dropout_key,
    )
    qf_state = qf_state.apply_gradients(grads=grads)
    qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])
    metrics = {
        "critic_loss": critic_loss,
        "current_q_values": current_q.mean(),
        "next_q_values": next_q.mean(),
    }
    return qf_state, metrics, key, current_q
