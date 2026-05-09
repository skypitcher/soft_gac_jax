from __future__ import annotations

import ast
import copy
import difflib
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from algorithms.dime_impl.scheduler import get_constant_schedule, get_cosine_schedule, get_linear_schedule

DIME_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = DIME_ROOT / "configs"
DYNAMIC_CONFIG_PREFIXES = {"env_kwargs", "tot_time_steps_by_env.overrides"}


class ConfigNode(dict):
    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{item}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self.wrap(value)

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{item}'") from exc

    def copy(self):
        return ConfigNode.wrap(dict(self))

    def log(self, logger) -> None:
        logger.info("─── Config ───────────────────────────────────────")
        text = config_to_text(self).strip()
        if text:
            for line in text.splitlines():
                logger.info("  %s", line)
        else:
            logger.info("  <empty>")
        logger.info("──────────────────────────────────────────────────")

    @classmethod
    def wrap(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls.wrap(v) for v in value)
        return value


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _auto_type(value: str):
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower == "none":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class _HelpRequested(Exception):
    pass


def _parse_overrides(argv: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    positional_alg_used = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {"-h", "--help"}:
            raise _HelpRequested
        if token.startswith("--"):
            item = token[2:]
            if "=" in item:
                key, value = item.split("=", 1)
                overrides[key] = _auto_type(value)
                i += 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith("--") and "=" not in argv[i + 1]:
                overrides[item] = _auto_type(argv[i + 1])
                i += 2
                continue
            overrides[item] = True
            i += 1
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            overrides[key] = _auto_type(value)
            i += 1
            continue
        if not positional_alg_used:
            overrides["alg"] = token
            positional_alg_used = True
            i += 1
            continue
        raise SystemExit(f"Unrecognized argument: {token}")
    return overrides


def _set_path(cfg: ConfigNode, path: str, value: Any) -> None:
    parts = path.split(".")
    current = cfg
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, Mapping):
            child = ConfigNode()
            current[part] = child
        elif not isinstance(child, ConfigNode):
            child = ConfigNode.wrap(child)
            current[part] = child
        current = child
    current[parts[-1]] = ConfigNode.wrap(value)


def _flatten_config_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        paths.append(path)
        if isinstance(value, Mapping):
            paths.extend(_flatten_config_paths(value, path))
    return paths


def _set_existing_path(cfg: ConfigNode, path: str, value: Any, valid_paths: list[str]) -> None:
    if path in DYNAMIC_CONFIG_PREFIXES or any(path.startswith(f"{prefix}.") for prefix in DYNAMIC_CONFIG_PREFIXES):
        _set_path(cfg, path, value)
        return

    parts = path.split(".")
    current: Mapping[str, Any] = cfg
    for part in parts[:-1]:
        child = current.get(part) if isinstance(current, Mapping) else None
        if not isinstance(child, Mapping):
            _raise_unknown_override(path, valid_paths)
        current = child
    if not isinstance(current, Mapping) or parts[-1] not in current:
        _raise_unknown_override(path, valid_paths)
    _set_path(cfg, path, value)


def _raise_unknown_override(path: str, valid_paths: list[str]) -> None:
    suggestion = difflib.get_close_matches(path, valid_paths, n=1, cutoff=0.45)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise SystemExit(f"Unknown config override '{path}'.{hint}")


def _instantiate_schedule(name: str, params: Mapping[str, Any], total_steps: int):
    name = str(name)
    params = dict(params)
    params.pop("_target_", None)
    params.pop("total_steps", None)
    if name == "cosine":
        return get_cosine_schedule(total_steps=total_steps, **params)
    if name == "linear":
        return get_linear_schedule(total_steps=total_steps, **params)
    if name == "constant":
        return get_constant_schedule()
    raise SystemExit(f"Unsupported dt schedule: {name}")


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _actor_num_steps(actor_cfg: Any) -> int:
    for key in ("diff_steps", "n_timesteps", "iter_steps", "num_layers"):
        value = getattr(actor_cfg, key, 0)
        if value:
            return int(value)
    return 0


def _env_step_lookup_keys(env_name: str) -> list[str]:
    keys: list[str] = []

    def add(value: str) -> None:
        if value and value not in keys:
            keys.append(value)

    add(env_name)
    if "/" in env_name:
        add(env_name.rsplit("/", 1)[-1])

    for key in list(keys):
        unversioned = re.sub(r"-v\d+$", "", key)
        if unversioned != key:
            add(unversioned)

    return keys


_GYM_MUJOCO_ENV_RE = re.compile(
    r"^(Ant|HalfCheetah|Hopper|Humanoid|HumanoidStandup|InvertedDoublePendulum|"
    r"InvertedPendulum|Pusher|Reacher|Swimmer|Walker2d)-v\d+$"
)


def _is_gym_mujoco_env(env_name: str) -> bool:
    return "/" not in env_name and _GYM_MUJOCO_ENV_RE.match(env_name) is not None


def _path_was_overridden(runtime_override_paths: set[str], path: str) -> bool:
    parts = path.split(".")
    for idx in range(1, len(parts) + 1):
        if ".".join(parts[:idx]) in runtime_override_paths:
            return True
    return False


def _resolve_tot_time_steps(cfg: ConfigNode, runtime_override_paths: set[str]) -> None:
    if "tot_time_steps" in runtime_override_paths:
        if cfg.tot_time_steps is None:
            raise SystemExit("tot_time_steps override must be an integer, not null.")
        cfg.tot_time_steps = int(cfg.tot_time_steps)
        cfg._tot_time_steps_source = "override:tot_time_steps"
        return

    steps_cfg = ConfigNode.wrap(cfg.get("tot_time_steps_by_env", {}))
    raw_tot_time_steps = cfg.get("tot_time_steps")
    fallback_steps = raw_tot_time_steps if raw_tot_time_steps is not None else 1_000_000
    default_steps = int(steps_cfg.get("default", fallback_steps))
    overrides = steps_cfg.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise SystemExit("tot_time_steps_by_env.overrides must be a mapping.")

    for key in _env_step_lookup_keys(str(cfg.env_name)):
        if key in overrides:
            cfg.tot_time_steps = int(overrides[key])
            cfg._tot_time_steps_source = f"tot_time_steps_by_env.overrides.{key}"
            return

    cfg.tot_time_steps = default_steps
    cfg._tot_time_steps_source = "tot_time_steps_by_env.default"


def _resolve_c51_support(cfg: ConfigNode, runtime_override_paths: set[str]) -> None:
    critic_cfg = cfg.get("alg", {}).get("critic") if isinstance(cfg.get("alg"), Mapping) else None
    if not isinstance(critic_cfg, Mapping):
        return
    if not _is_gym_mujoco_env(str(cfg.env_name)):
        return

    support_cfg = ConfigNode.wrap(cfg.get("c51_support_by_env", {})).get("gym_mujoco", {})
    support_cfg = ConfigNode.wrap(support_cfg)
    v_min = support_cfg.get("v_min", -1600)
    v_max = support_cfg.get("v_max", 1600)

    if "v_min" in critic_cfg and not _path_was_overridden(runtime_override_paths, "alg.critic.v_min"):
        cfg.alg.critic.v_min = v_min
    if "v_max" in critic_cfg and not _path_was_overridden(runtime_override_paths, "alg.critic.v_max"):
        cfg.alg.critic.v_max = v_max

    # FLAC's non-CrossQ path uses official_* support keys. Keep those aligned
    # with the Gym MuJoCo default unless the user explicitly overrides them.
    if "official_v_min" in critic_cfg and not _path_was_overridden(runtime_override_paths, "alg.critic.official_v_min"):
        cfg.alg.critic.official_v_min = v_min
    if "official_v_max" in critic_cfg and not _path_was_overridden(runtime_override_paths, "alg.critic.official_v_max"):
        cfg.alg.critic.official_v_max = v_max


def _finalize_config(cfg: ConfigNode, runtime_override_paths: set[str] | None = None) -> ConfigNode:
    runtime_override_paths = set(runtime_override_paths or ())
    _resolve_tot_time_steps(cfg, runtime_override_paths)
    _resolve_c51_support(cfg, runtime_override_paths)
    cfg.wandb.enabled = bool(cfg.wandb.enabled)
    cfg.step_size = float(cfg.alg.optimizer.lr_actor)
    cfg.step_size_betas = float(cfg.alg.optimizer.lr_actor)
    cfg.iters = int(cfg.tot_time_steps)

    if not cfg.wandb.get("job_type"):
        cfg.wandb.job_type = f"lr{cfg.alg.optimizer.lr_actor}"

    if "sampler" in cfg.alg:
        cfg.sampler = ConfigNode.wrap(cfg.alg.sampler)
        cfg.sampler.iters = int(cfg.iters)
        if "use_target_score" not in cfg.sampler or cfg.sampler.use_target_score is None or _is_placeholder(cfg.sampler.use_target_score):
            cfg.sampler.use_target_score = False

        if "score_model" in cfg.sampler:
            score_model = ConfigNode.wrap(cfg.sampler.score_model)
            if "use_target_score" not in score_model or score_model.use_target_score is None or _is_placeholder(score_model.use_target_score):
                score_model.use_target_score = bool(cfg.sampler.use_target_score)
            if "time_coder_out" not in score_model or score_model.time_coder_out is None or _is_placeholder(score_model.time_coder_out):
                score_model.time_coder_out = int(score_model.num_hid)
            cfg.sampler.score_model = score_model

        if "dt_schedule" in cfg.sampler:
            raw_schedule_cfg = dict(cfg.sampler.dt_schedule)
            raw_schedule_cfg.pop("_target_", None)
            total_steps = _actor_num_steps(cfg.alg.actor)
            raw_total_steps = raw_schedule_cfg.get("total_steps")
            if raw_total_steps is None or _is_placeholder(raw_total_steps):
                raw_schedule_cfg["total_steps"] = total_steps
            schedule_name = str(cfg.sampler.get("dt_schedule_name", "cosine"))
            runtime_schedule = _instantiate_schedule(schedule_name, raw_schedule_cfg, int(raw_schedule_cfg["total_steps"]))
            cfg.sampler.dt_schedule = runtime_schedule
            cfg.sampler._dt_schedule_cfg = ConfigNode.wrap(raw_schedule_cfg)

    cfg.algorithm = ConfigNode(
        {
            "num_steps": _actor_num_steps(cfg.alg.actor),
            "learn_betas": bool(getattr(getattr(cfg, "sampler", ConfigNode()), "learn_betas", False)),
            "target_score_max_norm": getattr(getattr(cfg, "sampler", ConfigNode()), "target_score_max_norm", None),
        }
    )
    return cfg


def _usage() -> str:
    return (
        "Usage: python main.py [alg] [key=value ...]\n"
        "Examples:\n"
        "  python main.py soft_gac env_name=dm_control/dog-run\n"
        "  python main.py alg=dime env_name=dm_control/humanoid-run alg.actor.diff_steps=8\n"
    )


def load_config(argv: list[str] | None = None) -> ConfigNode:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        overrides = _parse_overrides(argv)
    except _HelpRequested:
        print(_usage())
        raise SystemExit(0)

    main_cfg = _load_yaml_file(CONFIG_ROOT / "main.yaml")
    alg_name = str(overrides.get("alg", main_cfg.get("alg", "soft_gac")))
    merged = dict(main_cfg)
    merged["alg"] = _load_yaml_file(CONFIG_ROOT / "alg" / f"{alg_name}.yaml")

    cfg = ConfigNode.wrap(merged)
    runtime_overrides = {k: v for k, v in overrides.items() if k != "alg"}
    valid_paths = _flatten_config_paths(cfg)
    for path, value in runtime_overrides.items():
        _set_existing_path(cfg, path, value, valid_paths)

    cfg.alg.name = str(cfg.alg.get("name", alg_name))
    return _finalize_config(cfg, set(runtime_overrides))


def _serialize_value(value: Any):
    if isinstance(value, ConfigNode):
        return {k: _serialize_value(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, Mapping):
        return {k: _serialize_value(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__)
        return f"<callable:{name}>"
    return value


def config_to_plain_dict(cfg: ConfigNode) -> dict[str, Any]:
    return _serialize_value(cfg)


def config_to_text(cfg: ConfigNode) -> str:
    return yaml.safe_dump(config_to_plain_dict(cfg), sort_keys=False, allow_unicode=True)
