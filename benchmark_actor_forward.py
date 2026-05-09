import argparse
import csv
import gc
import os
import statistics
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from common.config import load_config
from algorithms.dime_impl.models.pisgrad_net import PISGRADNet
from models.actor import BridgePolicy, DenoisingNetwork, DiffusionDenoiser, FlowPolicy
from models.sac_policy import SquashedGaussianActor


TASK_DIMS: Dict[str, Tuple[int, int]] = {
    "dm_control/dog": (223, 38),
    "dm_control/humanoid": (67, 21),
    "dm_control/walker": (24, 6),
    "dog-run": (223, 38),
    "humanoid-run": (67, 21),
    "walker-run": (24, 6),
    "humanoid_bench": (51, 19),
    "h1": (51, 19),
}

DEFAULT_TASKS = ["dm_control/humanoid", "dm_control/dog", "dm_control/walker", "humanoid_bench"]
DEFAULT_METHODS = ["soft_gac", "dime", "flowrl", "flac", "qsm", "qvpo", "sac"]
METHOD_ALIASES = {"crossq_sac": "sac", "crossq-sac": "sac"}
METHOD_DISPLAY_NAMES = {
    "soft_gac": "SoftGAC",
    "dime": "DIME",
    "flowrl": "FlowRL",
    "flac": "FLAC",
    "qsm": "QSM",
    "qvpo": "QVPO",
    "sac": "CrossQ-SAC",
}
DEFAULT_SOFT_GAC_HIDDEN = 512
DOG_DOMAIN_SOFT_GAC_HIDDEN = 256
DEFAULT_METHOD_NFES = {
    "soft_gac": [1],
    "sac": [1],
    "dime": [16],
    "flowrl": [2],
    "flac": [2],
    "qsm": [5],
    "qvpo": [16],
}


@dataclass
class BenchCase:
    method: str
    nfe: int
    internal_steps: int
    actor_config: str
    params: dict
    forward_fn: object
    forward_args: tuple


def count_params(tree) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(tree))


def block_tree(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _timed_call_ms(fn, *args):
    start = time.perf_counter_ns()
    out = fn(*args)
    block_tree(out)
    end = time.perf_counter_ns()
    return (end - start) / 1e6


def _timed_chunk_mean_ms(fn, args, calls_per_chunk: int):
    start = time.perf_counter_ns()
    for _ in range(calls_per_chunk):
        out = fn(*args)
        block_tree(out)
    end = time.perf_counter_ns()
    total_ms = (end - start) / 1e6
    return total_ms / calls_per_chunk


def summarize_samples(samples_ms):
    return {
        "mean_ms": statistics.mean(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": sorted(samples_ms)[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))],
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def normalize_nfes(values) -> list[int]:
    nfes = sorted(set(int(n) for n in values))
    if any(n < 1 for n in nfes):
        raise SystemExit("NFE values must be positive integers.")
    return nfes


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        canonical = canonical_method(method)
        if canonical not in DEFAULT_METHODS:
            raise SystemExit(f"Unknown method: {method}")
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def parse_method_nfe_specs(specs) -> dict[str, list[int]]:
    method_nfes: dict[str, list[int]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"Invalid --method-nfes entry '{spec}'. Expected METHOD=NFE[,NFE...].")
        method, raw_values = spec.split("=", 1)
        method = canonical_method(method.strip())
        if method not in DEFAULT_METHODS:
            raise SystemExit(f"Unknown method in --method-nfes: {method}")
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise SystemExit(f"Missing NFE value for method {method}.")
        method_nfes[method] = normalize_nfes(values)
    return method_nfes


def validate_method_nfes(method: str, nfes: list[int]) -> list[int]:
    if method in {"soft_gac", "sac"}:
        if nfes != [1]:
            raise SystemExit(f"{method} is a one-pass actor; benchmark it with NFE=1.")
        return nfes

    if method in {"flowrl", "flac"}:
        valid = [n for n in nfes if n >= 2 and n % 2 == 0]
        invalid = [n for n in nfes if n not in valid]
        if invalid:
            print(f"warning: skipping {method} invalid NFE values {invalid}; midpoint flow needs even NFE >= 2.")
        if not valid:
            raise SystemExit(f"{method} requires at least one even NFE >= 2.")
        return valid

    return nfes


def resolve_method_nfes(args) -> dict[str, list[int]]:
    if args.global_nfe is not None and args.global_nfes is not None:
        raise SystemExit("Use only one of --nfe or --nfes.")

    method_nfes = {method: list(nfes) for method, nfes in DEFAULT_METHOD_NFES.items()}
    if args.global_nfe is not None:
        global_nfes = normalize_nfes([args.global_nfe])
    elif args.global_nfes is not None:
        global_nfes = normalize_nfes(args.global_nfes)
    else:
        global_nfes = None

    if global_nfes is not None:
        for method in DEFAULT_METHODS:
            if method not in {"soft_gac", "sac"}:
                method_nfes[method] = list(global_nfes)

    for method, nfes in parse_method_nfe_specs(args.method_nfes).items():
        method_nfes[method] = nfes

    named_overrides = {
        "dime": (args.dime_nfe, args.dime_nfes),
        "flowrl": (args.flowrl_nfe, args.flowrl_nfes),
        "flac": (args.flac_nfe, args.flac_nfes),
        "qsm": (args.qsm_nfe, args.qsm_nfes),
        "qvpo": (args.qvpo_nfe, args.qvpo_nfes),
    }
    for method, (single_nfe, nfes) in named_overrides.items():
        if single_nfe is not None and nfes is not None:
            raise SystemExit(f"Use only one of --{method}-nfe or --{method}-nfes.")
        if single_nfe is not None:
            method_nfes[method] = normalize_nfes([single_nfe])
        elif nfes is not None:
            method_nfes[method] = normalize_nfes(nfes)

    return {method: validate_method_nfes(method, method_nfes[method]) for method in DEFAULT_METHODS}


def format_method_nfes(method_nfes: dict[str, list[int]], methods: list[str]) -> str:
    return ", ".join(f"{method}={method_nfes[method]}" for method in methods)


def is_dog_domain(task: str) -> bool:
    return task in {"dm_control/dog", "dog-run"} or task.startswith("dm_control/dog-")


def resolve_soft_gac_hidden(task: str, args) -> int:
    if args.soft_gac_hidden is not None:
        return int(args.soft_gac_hidden)
    if is_dog_domain(task):
        return DOG_DOMAIN_SOFT_GAC_HIDDEN
    return DEFAULT_SOFT_GAC_HIDDEN


def format_soft_gac_defaults(args) -> str:
    if args.soft_gac_hidden is not None:
        return f"soft_gac(hidden={args.soft_gac_hidden}, K={args.soft_gac_k})"
    return (
        f"soft_gac(hidden={DEFAULT_SOFT_GAC_HIDDEN}, dog_hidden={DOG_DOMAIN_SOFT_GAC_HIDDEN}, "
        f"K={args.soft_gac_k})"
    )


def benchmark_one(
    fn,
    args,
    warmup: int,
    repeats: int,
    target_chunk_ms: float,
    calls_per_chunk: int | None = None,
):
    # First call triggers JAX compilation for this shape/function and is never timed.
    block_tree(fn(*args))
    for _ in range(warmup):
        block_tree(fn(*args))

    if calls_per_chunk is None:
        probe = statistics.median(_timed_call_ms(fn, *args) for _ in range(3))
        calls_per_chunk = max(1, ceil(target_chunk_ms / max(probe, 1e-6)))

    samples_ms = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            samples_ms.append(_timed_chunk_mean_ms(fn, args, calls_per_chunk))
    finally:
        if gc_was_enabled:
            gc.enable()

    return summarize_samples(samples_ms), calls_per_chunk


def build_local_actor(obs_dim: int, act_dim: int, hidden: int, k: int, dtype):
    action_scale = jnp.ones((act_dim,), dtype=dtype)
    action_bias = jnp.zeros((act_dim,), dtype=dtype)

    module = BridgePolicy(
        hidden_dim=hidden,
        num_actions=act_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        num_layers=k,
    )
    noise_shape = (1, k + 1, act_dim)

    return module, noise_shape


def build_dime_actor(act_dim: int, hidden: int, layers: int, use_target_score: bool):
    return PISGRADNet(
        dim=act_dim,
        use_target_score=use_target_score,
        num_hid=hidden,
        num_layers=layers,
        layer_norm=False,
        time_coder_out=hidden,
    )


def make_local_forward(module, params, batch_size, obs_dim, act_dim, k, dtype):
    obs = jnp.ones((batch_size, obs_dim), dtype=dtype)
    noise = jnp.ones((batch_size, k + 1, act_dim), dtype=dtype)

    @jax.jit
    def forward(obs_, noise_):
        return module.apply({"params": params}, obs_, noise_, method=module.sample)

    return forward, (obs, noise)


def make_dime_forward(module, params, batch_size, obs_dim, act_dim, nfe, dtype, use_target_score):
    obs = jnp.ones((batch_size, obs_dim), dtype=dtype)
    z0 = jnp.ones((batch_size, act_dim), dtype=dtype)
    if use_target_score:
        target_score = jnp.ones((batch_size, act_dim), dtype=dtype)
    else:
        target_score = None

    def one_step(carry, step_idx):
        t = jnp.full((batch_size, 1), step_idx / max(nfe, 1), dtype=dtype)
        if use_target_score:
            drift = module.apply({"params": params}, carry, obs, t, target_score)
        else:
            drift = module.apply({"params": params}, carry, obs, t)
        return carry + drift / max(nfe, 1), None

    @jax.jit
    def forward(z_):
        if nfe == 1:
            t = jnp.zeros((batch_size, 1), dtype=dtype)
            if use_target_score:
                return module.apply({"params": params}, z_, obs, t, target_score)
            return module.apply({"params": params}, z_, obs, t)
        z_final, _ = jax.lax.scan(one_step, z_, jnp.arange(nfe, dtype=jnp.int32))
        return z_final

    return forward, (z0,)


def make_diffusion_forward(module, params, batch_size, obs_dim, act_dim, nfe, dtype):
    obs = jnp.ones((batch_size, obs_dim), dtype=dtype)
    z0 = jnp.ones((batch_size, act_dim), dtype=dtype)

    def one_step(carry, step_idx):
        t = jnp.full((batch_size,), step_idx, dtype=jnp.int32)
        eps = module.apply({"params": params}, carry, t, obs)
        return carry - eps / max(nfe, 1), None

    @jax.jit
    def forward(z_):
        z_final, _ = jax.lax.scan(one_step, z_, jnp.arange(nfe, dtype=jnp.int32))
        return z_final

    return forward, (z0,)


def make_flow_forward(module, params, batch_size, obs_dim, act_dim, dtype):
    obs = jnp.ones((batch_size, obs_dim), dtype=dtype)
    noise = jnp.ones((batch_size, act_dim), dtype=dtype)

    @jax.jit
    def forward(obs_, noise_):
        return module.apply({"params": params}, obs_, noise=noise_, method=module.sample, train=False)

    return forward, (obs, noise)


def make_sac_forward(module, params, batch_stats, batch_size, obs_dim, dtype):
    obs = jnp.ones((batch_size, obs_dim), dtype=dtype)

    @jax.jit
    def forward(obs_):
        return module.apply({"params": params, "batch_stats": batch_stats}, obs_, method=module.deterministic, train=False)

    return forward, (obs,)


def build_bench_case(method: str, nfe: int, task: str, obs_dim: int, act_dim: int, args, dtype, key) -> BenchCase | None:
    if method == "soft_gac":
        soft_gac_hidden = resolve_soft_gac_hidden(task, args)
        module, noise_shape = build_local_actor(obs_dim, act_dim, soft_gac_hidden, args.soft_gac_k, dtype)
        obs = jnp.ones((1, obs_dim), dtype=dtype)
        noise = jnp.ones(noise_shape, dtype=dtype)
        params = module.init(key, obs, noise)["params"]
        forward, forward_args = make_local_forward(module, params, args.batch_size, obs_dim, act_dim, args.soft_gac_k, dtype)
        return BenchCase(method, 1, args.soft_gac_k, f"hidden={soft_gac_hidden},layers={args.soft_gac_k}", params, forward, forward_args)

    if method == "sac":
        cfg = load_config(["alg=sac"])
        module = SquashedGaussianActor(
            hidden_dim=int(cfg.alg.actor.hidden_size),
            num_layers=int(cfg.alg.actor.num_layers),
            num_actions=act_dim,
            action_scale=jnp.ones((act_dim,), dtype=dtype),
            action_bias=jnp.zeros((act_dim,), dtype=dtype),
            log_std_min=float(cfg.alg.actor.log_std_min),
            log_std_max=float(cfg.alg.actor.log_std_max),
            use_batch_norm=bool(cfg.alg.optimizer.bn),
            batch_norm_momentum=float(cfg.alg.optimizer.bn_momentum),
            batch_norm_mode=str(cfg.alg.optimizer.bn_mode),
            bn_warmup=int(cfg.alg.optimizer.bn_warmup),
        )
        obs = jnp.ones((1, obs_dim), dtype=dtype)
        variables = module.init({"params": key, "batch_stats": key}, obs, train=False)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", {})
        forward, forward_args = make_sac_forward(module, params, batch_stats, args.batch_size, obs_dim, dtype)
        return BenchCase(
            method,
            1,
            1,
            f"hidden={int(cfg.alg.actor.hidden_size)},layers={int(cfg.alg.actor.num_layers)}",
            params,
            forward,
            forward_args,
        )

    if method == "dime":
        module = build_dime_actor(act_dim, args.dime_hidden, args.dime_layers, args.dime_use_target_score)
        z = jnp.ones((1, act_dim), dtype=dtype)
        obs = jnp.ones((1, obs_dim), dtype=dtype)
        t = jnp.ones((1, 1), dtype=dtype)
        if args.dime_use_target_score:
            target_score = jnp.ones((1, act_dim), dtype=dtype)
            params = module.init(key, z, obs, t, target_score)["params"]
        else:
            params = module.init(key, z, obs, t)["params"]
        forward, forward_args = make_dime_forward(
            module, params, args.batch_size, obs_dim, act_dim, nfe, dtype, args.dime_use_target_score
        )
        return BenchCase(method, nfe, nfe, f"hidden={args.dime_hidden},layers={args.dime_layers}", params, forward, forward_args)

    if method in {"flowrl", "flac"}:
        if nfe < 2 or nfe % 2 != 0:
            return None
        cfg = load_config([f"alg={method}"])
        steps = nfe // 2
        module = FlowPolicy(
            hidden_dim=int(cfg.alg.actor.hidden_size),
            num_actions=act_dim,
            steps=steps,
            action_scale=jnp.ones((act_dim,), dtype=dtype),
            action_bias=jnp.zeros((act_dim,), dtype=dtype),
            num_layers=int(cfg.alg.actor.num_layers),
        )
        obs = jnp.ones((1, obs_dim), dtype=dtype)
        action = jnp.ones((1, act_dim), dtype=dtype)
        t = jnp.zeros((1, 1), dtype=dtype)
        params = module.init(key, obs, action, t, train=False)["params"]
        forward, forward_args = make_flow_forward(module, params, args.batch_size, obs_dim, act_dim, dtype)
        return BenchCase(
            method,
            nfe,
            steps,
            f"hidden={int(cfg.alg.actor.hidden_size)},layers={int(cfg.alg.actor.num_layers)}",
            params,
            forward,
            forward_args,
        )

    if method == "qsm":
        cfg = load_config(["alg=qsm"])
        module = DenoisingNetwork(
            hidden_dim=int(cfg.alg.actor.hidden_size),
            action_dim=act_dim,
            time_embed_dim=int(cfg.alg.actor.time_embed_dim),
            num_layers=int(cfg.alg.actor.num_layers),
        )
    elif method == "qvpo":
        cfg = load_config([f"alg={method}"])
        module = DiffusionDenoiser(
            hidden_dim=int(cfg.alg.actor.hidden_size),
            action_dim=act_dim,
            time_embed_dim=int(cfg.alg.actor.time_embed_dim),
            num_layers=int(cfg.alg.actor.num_layers),
            use_layer_norm=bool(cfg.alg.actor.use_layer_norm),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    z = jnp.ones((1, act_dim), dtype=dtype)
    obs = jnp.ones((1, obs_dim), dtype=dtype)
    t = jnp.zeros((1,), dtype=jnp.int32)
    params = module.init(key, z, t, obs)["params"]
    forward, forward_args = make_diffusion_forward(module, params, args.batch_size, obs_dim, act_dim, nfe, dtype)
    return BenchCase(
        method,
        nfe,
        nfe,
        f"hidden={int(cfg.alg.actor.hidden_size)},layers={int(cfg.alg.actor.num_layers)}",
        params,
        forward,
        forward_args,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark actor inference time on fake inputs.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=sorted(TASK_DIMS))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(set(DEFAULT_METHODS) | set(METHOD_ALIASES)),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--target-chunk-ms",
        "--target-block-ms",
        dest="target_chunk_ms",
        type=float,
        default=200.0,
        help="Choose calls_per_chunk automatically so each timed chunk is roughly this long.",
    )
    parser.add_argument(
        "--calls-per-chunk",
        "--inner-iters",
        dest="calls_per_chunk",
        type=int,
        default=None,
        help="Override the number of consecutive forward calls per timed chunk.",
    )
    parser.add_argument("--local-inner-iters", dest="calls_per_chunk", type=int)
    parser.add_argument("--dime-inner-iters", dest="calls_per_chunk", type=int)
    parser.add_argument(
        "--soft-gac-hidden",
        "--local-hidden",
        dest="soft_gac_hidden",
        type=int,
        default=None,
        help="Override SoftGAC actor hidden size for all tasks. By default dog domain uses 256 and others use 512.",
    )
    parser.add_argument("--soft-gac-k", "--local-k", dest="soft_gac_k", type=int, default=6)
    parser.add_argument("--dime-hidden", type=int, default=256)
    parser.add_argument("--dime-layers", type=int, default=3)
    parser.add_argument("--nfe", dest="global_nfe", type=int, default=None)
    parser.add_argument("--nfes", dest="global_nfes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--method-nfes",
        nargs="+",
        default=None,
        metavar="METHOD=NFE[,NFE...]",
        help="Override NFE per method, e.g. --method-nfes dime=16 qsm=5 flowrl=2.",
    )
    parser.add_argument("--dime-nfe", type=int, default=None)
    parser.add_argument("--dime-nfes", type=int, nargs="+", default=None)
    parser.add_argument("--flowrl-nfe", type=int, default=None)
    parser.add_argument("--flowrl-nfes", type=int, nargs="+", default=None)
    parser.add_argument("--flac-nfe", type=int, default=None)
    parser.add_argument("--flac-nfes", type=int, nargs="+", default=None)
    parser.add_argument("--qsm-nfe", type=int, default=None)
    parser.add_argument("--qsm-nfes", type=int, nargs="+", default=None)
    parser.add_argument("--qvpo-nfe", type=int, default=None)
    parser.add_argument("--qvpo-nfes", type=int, nargs="+", default=None)
    parser.add_argument("--dime-use-target-score", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    args.methods = normalize_methods(args.methods)

    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    method_nfes = resolve_method_nfes(args)

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        "env: "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')} "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '<unset>')}"
    )
    print(
        f"benchmark: methods={args.methods} nfes=({format_method_nfes(method_nfes, args.methods)}) "
        f"{format_soft_gac_defaults(args)} "
        f"dime(hidden={args.dime_hidden}, layers={args.dime_layers})"
    )
    print(
        f"batch_size={args.batch_size} warmup={args.warmup} repeats={args.repeats} "
        f"dtype={args.dtype} target_chunk_ms={args.target_chunk_ms:.1f} compile_time=excluded"
    )
    print()

    rows = []
    for task in args.tasks:
        obs_dim, act_dim = TASK_DIMS[task]
        print(f"[{task}] obs={obs_dim} act={act_dim}")
        task_rows = []
        for method_idx, method in enumerate(args.methods):
            for nfe in method_nfes[method]:
                key = jax.random.PRNGKey(method_idx * 1000 + nfe)
                case = build_bench_case(method, nfe, task, obs_dim, act_dim, args, dtype, key)
                if case is None:
                    continue

                stats, calls_per_chunk = benchmark_one(
                    case.forward_fn,
                    case.forward_args,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    target_chunk_ms=args.target_chunk_ms,
                    calls_per_chunk=args.calls_per_chunk,
                )
                params = count_params(case.params)
                row = {
                    "task": task,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "method": case.method,
                    "algorithm": METHOD_DISPLAY_NAMES[case.method],
                    "nfe": case.nfe,
                    "internal_steps": case.internal_steps,
                    "actor_config": case.actor_config,
                    "params_k": f"{params / 1000:.1f}",
                    "batch_size": args.batch_size,
                    "backend": jax.default_backend(),
                    "compile_time_excluded": True,
                    "calls_per_chunk": calls_per_chunk,
                    "timed_forward_calls": args.repeats * calls_per_chunk,
                    "mean_batch_ms": f"{stats['mean_ms']:.6f}",
                    "p50_batch_ms": f"{stats['p50_ms']:.6f}",
                    "p95_batch_ms": f"{stats['p95_ms']:.6f}",
                    "mean_per_action_us": f"{stats['mean_ms'] * 1000 / args.batch_size:.6f}",
                }
                rows.append(row)
                task_rows.append(row)
                print(
                    f"  {case.method:8s} nfe={case.nfe:<2d} steps={case.internal_steps:<2d} "
                    f"cfg={case.actor_config:<22s} "
                    f"params={params / 1000:7.1f}K "
                    f"mean={stats['mean_ms']:.4f} ms/batch "
                    f"per_action={stats['mean_ms'] * 1000 / args.batch_size:.4f} us "
                    f"p95={stats['p95_ms']:.4f} calls_per_chunk={calls_per_chunk}"
                )

        if task_rows:
            soft_rows = [r for r in task_rows if r["method"] == "soft_gac"]
            if soft_rows:
                soft_ms = float(soft_rows[0]["mean_batch_ms"])
                for row in task_rows:
                    row["speedup_vs_soft_gac"] = f"{float(row['mean_batch_ms']) / soft_ms:.4f}"
        print()

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "task",
            "obs_dim",
            "act_dim",
            "method",
            "algorithm",
            "nfe",
            "internal_steps",
            "actor_config",
            "params_k",
            "batch_size",
            "backend",
            "compile_time_excluded",
            "calls_per_chunk",
            "timed_forward_calls",
            "mean_batch_ms",
            "p50_batch_ms",
            "p95_batch_ms",
            "mean_per_action_us",
            "speedup_vs_soft_gac",
        ]
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
