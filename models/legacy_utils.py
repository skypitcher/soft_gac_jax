import jax
import jax.numpy as jnp


def soft_update(target_params, source_params, tau):
    """Exponential moving average: target = (1 - tau) * target + tau * source."""
    return jax.tree.map(lambda t, s: t * (1.0 - tau) + s * tau, target_params, source_params)


def hard_update(target_params, source_params):
    """Direct copy: target = source."""
    return source_params


def barycentric_ot_1d_searchsorted(u_sorted, v_sorted, w_sorted, num_particles):
    """O(N) barycentric OT projection via target CDF searchsorted.

    Source weights are uniform 1/N. This computes the exact integral of the
    weighted target quantile function over each source bin.
    """
    weights_cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w_sorted)])
    source_edges = jnp.arange(num_particles + 1) / num_particles

    target_idx = jnp.searchsorted(weights_cdf, source_edges, side="right") - 1
    target_idx = jnp.clip(target_idx, 0, num_particles - 1)

    weighted_values = v_sorted * w_sorted
    primitive_prefix = jnp.concatenate([jnp.zeros(1), jnp.cumsum(weighted_values)])
    primitive_at_edges = (
        primitive_prefix[target_idx]
        + v_sorted[target_idx] * (source_edges - weights_cdf[target_idx])
    )
    return num_particles * (primitive_at_edges[1:] - primitive_at_edges[:-1])


def mish(x):
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(jax.nn.softplus(x))


def sinusoidal_embedding(t, embed_dim):
    """Sinusoidal positional embedding for diffusion timestep."""
    t = t.reshape(-1).astype(jnp.float32)
    half_dim = max(1, embed_dim // 2)
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if emb.shape[-1] > embed_dim:
        emb = emb[:, :embed_dim]
    elif emb.shape[-1] < embed_dim:
        emb = jnp.pad(emb, ((0, 0), (0, embed_dim - emb.shape[-1])))
    return emb


def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule for DDPM diffusion variances."""
    steps = jnp.arange(T + 1, dtype=jnp.float32)
    f = jnp.cos(((steps / T) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod_full = f / f[0]
    betas = 1.0 - alphas_cumprod_full[1:] / alphas_cumprod_full[:-1]
    betas = jnp.clip(betas, 0.0, 0.999)
    alphas_cumprod = jnp.cumprod(1.0 - betas)
    return betas, alphas_cumprod


def vp_beta_schedule(T):
    """Variance-preserving schedule used by the official QSM code."""
    t = jnp.arange(1, T + 1, dtype=jnp.float32)
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2.0 * t - 1.0) / (T ** 2))
    betas = 1.0 - alpha
    betas = jnp.clip(betas, 0.0, 0.999)
    alphas_cumprod = jnp.cumprod(1.0 - betas)
    return betas, alphas_cumprod


def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """Linear schedule kept for completeness / ablation."""
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    betas = jnp.clip(betas, 0.0, 0.999)
    alphas_cumprod = jnp.cumprod(1.0 - betas)
    return betas, alphas_cumprod


def inverse_softplus(x):
    """Numerically stable inverse of softplus: log(exp(x) - 1)."""
    threshold = 20.0
    return jnp.where(x > threshold, x, jnp.log(jnp.expm1(x)))


def project_c51_target(next_probs, reward, mask, gamma, z_atoms, v_min, v_max):
    """Project scalar Bellman targets onto a fixed C51 support."""
    num_atoms = z_atoms.shape[0]
    delta_z = (v_max - v_min) / (num_atoms - 1)
    target_z = reward + mask * gamma * z_atoms[None, :]
    target_z = jnp.clip(target_z, v_min, v_max)

    b = (target_z - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)
    l = jnp.clip(l, 0, num_atoms - 1)
    u = jnp.clip(u, 0, num_atoms - 1)

    wl = u.astype(jnp.float32) - b
    wu = b - l.astype(jnp.float32)
    eq = (u == l)
    wl = jnp.where(eq, 1.0, wl)
    wu = jnp.where(eq, 0.0, wu)

    target = jnp.zeros((reward.shape[0], num_atoms), dtype=jnp.float32)
    batch_idx = jnp.arange(reward.shape[0])[:, None]
    target = target.at[batch_idx, l].add(next_probs * wl)
    target = target.at[batch_idx, u].add(next_probs * wu)
    return target


def normal_log_prob(x, mean, scale):
    """Log probability of x under independent Normal(mean, scale)."""
    var = scale ** 2
    log_scale = jnp.log(scale)
    return -0.5 * jnp.sum(((x - mean) ** 2) / var + 2.0 * log_scale + jnp.log(2.0 * jnp.pi))


def normal_sample(key, mean, scale):
    """Sample from Normal(mean, scale)."""
    eps = jax.random.normal(key, shape=mean.shape)
    return mean + scale * eps


def tanh_log_det_jacobian(x):
    """Log |det J| for tanh transform, summed over dimensions."""
    return jnp.sum(2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x)))


def mvn_log_prob(x, mean, std):
    """Log prob of multivariate diagonal normal (summed over dims)."""
    return normal_log_prob(x, mean, std)


def mvn_sample(key, mean, std, n_samples):
    """Sample n_samples from MultivariateNormalDiag(mean, std)."""
    eps = jax.random.normal(key, shape=(n_samples,) + mean.shape)
    return mean + std * eps
