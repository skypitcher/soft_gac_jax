import os
import sys

# Avoid GLFW DISPLAY warnings on headless Linux machines.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL") in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import gymnasium as gym

HUMANOID_BENCH_PREFIXES = (
    "h1-",
    "h1hand-",
    "h1simplehand-",
    "h1strong-",
    "h1touch-",
    "g1-",
    "digit-",
)
HUMANOID_BENCH_STRING_BOOL_KWARGS = {"blocked_hands", "obs_wrapper", "small_obs"}


def _looks_like_humanoid_bench(env_name: str) -> bool:
    return str(env_name).startswith(HUMANOID_BENCH_PREFIXES)


def _register_optional_envs(env_name: str) -> None:
    if not _looks_like_humanoid_bench(env_name):
        return

    try:
        import humanoid_bench  # noqa: F401
    except ImportError as exc:
        if getattr(exc, "name", None) == "humanoid_bench":
            raise RuntimeError(
                "HumanoidBench env requested but 'humanoid_bench' is not installed. "
                "Use the dedicated setup path so it is installed editable with "
                "mujoco==3.1.6 and without shimmy."
            ) from exc
        raise RuntimeError(
            "HumanoidBench import failed after installation. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "HumanoidBench import failed after installation. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _normalize_env_kwargs(env_name: str, env_kwargs: dict | None) -> dict:
    normalized = dict(env_kwargs or {})
    if _looks_like_humanoid_bench(env_name):
        for key in HUMANOID_BENCH_STRING_BOOL_KWARGS:
            value = normalized.get(key)
            if isinstance(value, bool):
                normalized[key] = "True" if value else "False"
    return normalized


def make_env(env_name: str, seed: int | None = None, env_kwargs: dict | None = None):
    _register_optional_envs(env_name)
    env = gym.make(env_name, **_normalize_env_kwargs(env_name, env_kwargs))
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
    return env


def make_eval_env(env_name: str, seed: int, env_kwargs: dict | None = None):
    from stable_baselines3.common.env_util import make_vec_env

    _register_optional_envs(env_name)
    return make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs=_normalize_env_kwargs(env_name, env_kwargs))
