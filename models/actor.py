import flax.linen as nn
import jax
import jax.numpy as jnp

from models.legacy_utils import mish, sinusoidal_embedding

xavier_init = nn.initializers.xavier_uniform()
zero_init = nn.initializers.zeros
bridge_scale_init = nn.initializers.constant(0.541324854612918)


class BridgePolicy(nn.Module):
    """Time-scaled stochastic bridge actor used by SoftGAC.

    The bridge is written on a unit pseudo-time horizon with uniform step size
    ``h = 1 / num_layers``:

    ``z_{k+1} = z_k + h * drift_k(s, z_k) + sqrt(2h) * sigma_k(s, z_k) * eps_k``.
    The first noise slice is the pre-sampled base latent ``z_0``.
    """

    hidden_dim: int
    num_actions: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    num_layers: int = 2
    base_distribution: str = "logistic"

    def _run_bridge(self, state, noise):
        if noise.ndim != 3:
            raise ValueError("BridgePolicy expects noise with shape [batch, num_layers + 1, num_actions].")
        if noise.shape[1] != self.num_layers + 1:
            raise ValueError("BridgePolicy noise must provide a0 plus one independent noise tensor per step.")
        if noise.shape[2] != self.num_actions:
            raise ValueError("BridgePolicy noise last dimension must match num_actions.")

        h = jnp.asarray(1.0 / self.num_layers, dtype=state.dtype)
        mean_scale = h
        noise_scale = jnp.sqrt(2.0 * h)

        action = noise[:, 0, :]
        latents = [action]
        drifts = []
        sigmas = []
        for i in range(self.num_layers):
            step_noise = noise[:, i + 1, :]
            step_in = jnp.concatenate([state, action], axis=-1)
            step_in = nn.LayerNorm(name=f"step_{i}_input_ln")(step_in)
            hidden = nn.Dense(
                self.hidden_dim,
                kernel_init=xavier_init,
                bias_init=zero_init,
                name=f"step_{i}_hidden",
            )(step_in)
            hidden = nn.elu(hidden)
            hidden = nn.LayerNorm(name=f"step_{i}_hidden_ln")(hidden)
            drift = nn.Dense(
                self.num_actions,
                kernel_init=xavier_init,
                bias_init=zero_init,
                name=f"drift_{i}_out",
            )(hidden)
            scale_logits = nn.Dense(
                self.num_actions,
                kernel_init=zero_init,
                bias_init=bridge_scale_init,
                name=f"scale_{i}_out",
            )(hidden)
            sigma = nn.softplus(scale_logits)
            drifts.append(drift)
            sigmas.append(sigma)
            action = action + mean_scale * drift + noise_scale * sigma * step_noise
            latents.append(action)
        latents = jnp.stack(latents, axis=1)
        drifts = jnp.stack(drifts, axis=1)
        sigmas = jnp.stack(sigmas, axis=1)
        return action, latents, drifts, sigmas

    @nn.compact
    def __call__(self, state, noise, *, train: bool = False):
        action, _, _, _ = self._run_bridge(state, noise)
        return action

    @nn.compact
    def path_stats(self, state, noise, *, train: bool = False):
        return self._run_bridge(state, noise)

    def squash(self, z):
        return jnp.tanh(z) * self.action_scale + self.action_bias

    def sample(self, state, noise=None, *, rng=None, train: bool = False):
        if noise is None:
            base_key, step_key = jax.random.split(rng)
            if self.base_distribution == "logistic":
                eps = jnp.asarray(1e-6, dtype=state.dtype)
                base_action = jax.random.uniform(
                    base_key,
                    (state.shape[0], self.num_actions),
                    minval=-1.0 + eps,
                    maxval=1.0 - eps,
                    dtype=state.dtype,
                )
                base_latent = jnp.arctanh(base_action)
            elif self.base_distribution == "normal":
                base_latent = jax.random.normal(
                    base_key,
                    (state.shape[0], self.num_actions),
                    dtype=state.dtype,
                )
            else:
                raise ValueError(f"Unsupported BridgePolicy base_distribution: {self.base_distribution}")
            step_noise = jax.random.normal(
                step_key,
                (state.shape[0], self.num_layers, self.num_actions),
                dtype=state.dtype,
            )
            noise = jnp.concatenate([base_latent[:, None, :], step_noise], axis=1)
        return self.squash(self(state, noise, train=train))


class DenoisingNetwork(nn.Module):
    """DDPM denoising / score network with sinusoidal time embedding."""

    hidden_dim: int
    action_dim: int
    time_embed_dim: int = 64
    num_layers: int = 3
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x_t, t, state):
        h = self.hidden_dim
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = nn.Dense(h, kernel_init=xavier_init, bias_init=zero_init)(t_emb)
        t_emb = mish(t_emb)

        x = jnp.concatenate([state, x_t, t_emb], axis=-1)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="input_ln")(x)
        for _ in range(self.num_layers):
            x = nn.Dense(h, kernel_init=xavier_init, bias_init=zero_init)(x)
            x = mish(x)
        return nn.Dense(self.action_dim, kernel_init=xavier_init, bias_init=zero_init)(x)


class DiffusionDenoiser(nn.Module):
    """DDPM denoiser used by diffusion-policy baselines."""

    hidden_dim: int
    action_dim: int
    time_embed_dim: int = 32
    num_layers: int = 3
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x_t, t, state):
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = nn.Dense(self.hidden_dim, kernel_init=xavier_init, bias_init=zero_init)(t_emb)
        t_emb = mish(t_emb)
        t_emb = nn.Dense(self.time_embed_dim, kernel_init=xavier_init, bias_init=zero_init)(t_emb)

        x = jnp.concatenate([state, x_t, t_emb], axis=-1)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="input_ln")(x)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=xavier_init, bias_init=zero_init, name=f"hidden_{i}")(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"hidden_{i}_ln")(x)
            x = mish(x)
        return nn.Dense(self.action_dim, kernel_init=xavier_init, bias_init=zero_init, name="out")(x)


class ScoreNetwork(nn.Module):
    """DIME score network with learned Fourier time embedding."""

    dim: int
    num_hid: int = 64
    num_layers: int = 2
    time_coder_out: int = 64
    use_layer_norm: bool = False
    inner_clip: float = 1e2
    outer_clip: float = 1e4
    use_target_score: bool = False
    weight_init: float = 1e-8
    bias_init: float = 0.0

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase",
            nn.initializers.zeros_init(),
            (1, self.num_hid),
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100.0, num=self.num_hid)[None]
        self.time_coder_state = nn.Sequential(
            [
                nn.Dense(self.num_hid),
                nn.gelu,
                nn.Dense(self.time_coder_out),
            ]
        )

        if self.use_layer_norm:
            layers = [nn.Sequential([nn.Dense(self.num_hid), nn.LayerNorm(), nn.gelu]) for _ in range(self.num_layers)]
        else:
            layers = [nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)]
        layers.append(
            nn.Dense(
                self.dim,
                kernel_init=nn.initializers.constant(self.weight_init),
                bias_init=nn.initializers.constant(self.bias_init),
            )
        )
        self.state_time_net = nn.Sequential(layers)

        if self.use_target_score:
            grad_layers = [nn.Dense(self.num_hid)]
            grad_layers.extend([nn.Sequential([nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)])
            grad_layers.append(
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(self.weight_init),
                    bias_init=nn.initializers.constant(self.bias_init),
                )
            )
            self.time_coder_grad = nn.Sequential(grad_layers)

    def get_fourier_features(self, timesteps):
        sin_embed = jnp.sin(self.timestep_coeff * timesteps + self.timestep_phase)
        cos_embed = jnp.cos(self.timestep_coeff * timesteps + self.timestep_phase)
        return jnp.concatenate([sin_embed, cos_embed], axis=-1)

    def __call__(self, input_array, obs_array, time_array, target_score=None):
        time_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_emb = time_emb[0]

        t_state = self.time_coder_state(time_emb)
        extended_input = jnp.concatenate((input_array, obs_array, t_state), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        if not self.use_target_score:
            return out_state

        t_grad = self.time_coder_grad(time_emb)
        target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
        return out_state + t_grad * target_score


class FlowPolicy(nn.Module):
    """ODE-based flow policy with midpoint integration."""

    hidden_dim: int
    num_actions: int
    steps: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    num_layers: int = 2

    @nn.compact
    def __call__(self, state, action_0, time, *, train: bool = False):
        """Velocity field v(s, a, t)."""
        x = jnp.concatenate([state, action_0, time], axis=-1)
        for i in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=xavier_init,
                bias_init=zero_init,
                name=f"hidden_{i}",
            )(x)
            x = nn.LayerNorm(name=f"hidden_{i}_ln")(x)
            x = nn.elu(x)
        return nn.Dense(
            self.num_actions,
            kernel_init=xavier_init,
            bias_init=zero_init,
            name="out",
        )(x)

    def step(self, state, action, time_start, time_end, *, train: bool = False):
        """Midpoint integration from time_start to time_end."""
        dt = time_end - time_start
        v_start = self(state, action, time_start, train=train)
        mid_action = action + v_start * dt / 2
        v_mid = self(state, mid_action, time_start + dt / 2, train=train)
        return action + v_mid * dt

    def integrate(self, state, noise=None, *, rng=None, train: bool = False):
        """ODE integration, returns pre-tanh output z."""
        time_start = jnp.zeros((state.shape[0], 1))
        time_step = 1.0 / self.steps
        if noise is None:
            action = jax.random.normal(rng, (state.shape[0], self.num_actions))
            action = jnp.clip(action, -1.0, 1.0)
        else:
            action = noise

        for _ in range(self.steps):
            time_end = time_start + time_step
            action = self.step(state, action, time_start, time_end, train=train)
            time_start = time_end

        return action

    def squash(self, z):
        return jnp.tanh(z) * self.action_scale + self.action_bias

    def sample(self, state, noise=None, *, rng=None, train: bool = False):
        return self.squash(self.integrate(state, noise=noise, rng=rng, train=train))

    def sample_with_kinetic(self, state, *, rng, train: bool = False):
        """Sample action and accumulate kinetic energy along the trajectory."""
        time_start = jnp.zeros((state.shape[0], 1))
        time_step = 1.0 / self.steps
        action = jax.random.normal(rng, (state.shape[0], self.num_actions))
        action = jnp.clip(action, -1.0, 1.0)

        total_kinetic = jnp.zeros((state.shape[0], 1))
        for _ in range(self.steps):
            time_end = time_start + time_step
            dt = time_end - time_start
            v_start = self(state, action, time_start, train=train)
            mid_action = action + v_start * dt / 2
            v_mid = self(state, mid_action, time_start + dt / 2, train=train)
            action = action + v_mid * dt
            total_kinetic = total_kinetic + 0.5 * (v_mid**2).sum(axis=-1, keepdims=True) * dt
            time_start = time_end

        return self.squash(action), total_kinetic
