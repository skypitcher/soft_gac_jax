from typing import Sequence, Optional, Type, Callable, Any, Union, List, Dict, Tuple
import jax
import numpy as np

import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.linen import initializers
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from common.policies import BaseJaxPolicy
from common.type_aliases import ActorTrainState, RLTrainState
from models.legacy_utils import mish

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
  and adapted from Flax's BatchNorm implementation:
  https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


  Attributes:
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    bn_warmup: int = 100_000
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r ** 2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                # warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                warmed_up = jnp.greater_equal(steps.value, self.bn_warmup).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                        self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    bn_warmup: int = 100_000
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    bn_mode: str = "bn"
    n_atoms: int = 101

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:
        if 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        elif 'brn' in self.bn_mode:
            BN = BatchRenorm
        else:
            raise NotImplementedError

        x = jnp.concatenate([x, action], -1)

        if self.use_batch_norm:
            x = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation_fn()(x)

            if self.use_batch_norm:
                x = BN(bn_warmup=self.bn_warmup,use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        x = nn.Dense(self.n_atoms)(x)
        # x = nn.Dense(1, kernel_init=nn.initializers.constant(1e-6),
        #                 bias_init=nn.initializers.constant(0.0))(x)
        if self.n_atoms > 1:
            x = jax.nn.softmax(x, axis=-1)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    bn_warmup: int = 100_000
    use_batch_norm: bool = False
    batch_norm_mode: str = "bn"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    n_atoms: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_warmup=self.bn_warmup,
            bn_mode=self.batch_norm_mode,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            n_atoms=self.n_atoms
        )(obs, action, train)
        return q_values


class FLACOfficialQHead(nn.Module):
    """Q head matching the official FLAC critic block."""

    hidden_dim: int
    block_scale: int = 4
    output_dim: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="input",
        )(x)

        inner = self.hidden_dim * self.block_scale
        x = nn.LayerNorm(name="block_ln0")(x)
        x = nn.Dense(
            inner,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="block_dense0",
        )(x)
        x = nn.LayerNorm(name="block_ln1")(x)
        x = nn.gelu(x)
        x = nn.Dense(
            inner,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="block_dense1",
        )(x)
        x = nn.LayerNorm(name="block_ln2")(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="block_dense2",
        )(x)
        x = nn.LayerNorm(name="block_ln3")(x)
        return nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="value",
        )(x)


class FLACOfficialCritic(nn.Module):
    """Official FLAC-style twin critic.

    The architecture mirrors the official PyTorch `QNetwork`: each Q head has
    an input projection, a layer-normalized value block, and a value/logit
    output. With ``n_atoms > 1`` this is the official C51 critic path.
    """

    hidden_dim: int
    block_scale: int = 4
    n_critics: int = 2
    n_atoms: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        q_values = [
            FLACOfficialQHead(
                hidden_dim=self.hidden_dim,
                block_scale=self.block_scale,
                output_dim=self.n_atoms,
                name=f"q{i + 1}",
            )(obs, action)
            for i in range(self.n_critics)
        ]
        return jnp.stack(q_values, axis=0)


class FlowRLValueNet(nn.Module):
    """Layer-normalized MLP used by FlowRL's auxiliary value network."""

    hidden_dim: int
    output_dim: int
    num_layers: int
    activation: str

    def _act(self, x):
        if self.activation == "mish":
            return mish(x)
        if self.activation == "gelu":
            return nn.gelu(x)
        if self.activation == "elu":
            return nn.elu(x)
        if self.activation == "relu":
            return nn.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    @nn.compact
    def __call__(self, x):
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {self.num_layers}")

        for i in range(self.num_layers):
            x = nn.LayerNorm(name=f"hidden_{i}_ln")(x)
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name=f"hidden_{i}",
            )(x)
            x = self._act(x)

        x = nn.LayerNorm(name="final_ln")(x)
        return nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="out",
        )(x)


class FlowRLBufferCritic(nn.Module):
    """Layer-normalized twin critic used by FlowRL's auxiliary buffer critic."""

    hidden_dim: int
    critic_num_layers: int = 2
    activation: str = "gelu"
    bandwidth_scale: int = 1

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)

        def value_block(h, prefix):
            if self.critic_num_layers < 1:
                raise ValueError("FlowRLBufferCritic requires critic_num_layers >= 1")

            inner = self.hidden_dim * self.bandwidth_scale
            for i in range(self.critic_num_layers):
                h = nn.LayerNorm(name=f"{prefix}_ln{i}")(h)
                if i > 0:
                    if self.activation == "gelu":
                        h = nn.gelu(h)
                    elif self.activation == "silu":
                        h = nn.silu(h)
                    elif self.activation == "relu":
                        h = nn.relu(h)
                    elif self.activation == "elu":
                        h = nn.elu(h)
                    else:
                        raise ValueError(f"Unsupported FlowRLBufferCritic activation: {self.activation}")
                out_dim = self.hidden_dim if i == self.critic_num_layers - 1 else inner
                h = nn.Dense(
                    out_dim,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.zeros,
                    name=f"{prefix}_dense{i}",
                )(h)
            h = nn.LayerNorm(name=f"{prefix}_ln{self.critic_num_layers}")(h)
            return h

        q1 = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="q1_input",
        )(x)
        q1 = value_block(q1, "q1_block")
        q1 = nn.Dense(
            1,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="q1_value",
        )(q1)

        q2 = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="q2_input",
        )(x)
        q2 = value_block(q2, "q2_block")
        q2 = nn.Dense(
            1,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="q2_value",
        )(q2)
        return q1, q2
