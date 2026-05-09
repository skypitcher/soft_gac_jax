import jax
import jax.numpy as jnp

from algorithms.dime_impl.utils import sample_kernel, log_prob_kernel, check_stop_grad


def get_integrator(cfg, diffusion_model):
    def integrator(model_state, params, obs, stop_grad=False):

        def integrate_EM(state, per_step_input):
            x, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            sigma_square = 1. / diffusion_model.friction_fn(step, params)
            eta = dt * sigma_square
            scale = jnp.sqrt(2 * eta)

            # Forward kernel
            drift, aux = diffusion_model.drift_fn(step, x, params)
            fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs, model_state, params, aux))
            key, key_gen = jax.random.split(key_gen)
            x_new = sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)

            # Backward kernel
            drift_new, aux_new = diffusion_model.drift_fn(step + 1, x_new, params)
            bwd_mean = x_new + eta * (
                    drift_new + diffusion_model.backward_model(step + 1, x_new, obs, model_state, params, aux_new))

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, log_w, key_gen)
            per_step_output = x_new
            return next_state, per_step_output

        if cfg.sampler.integrator == 'EM':
            integrate = integrate_EM
        else:
            raise ValueError(f'No integrator named {cfg.sampler.integrator}.')

        return integrate

    return integrator
