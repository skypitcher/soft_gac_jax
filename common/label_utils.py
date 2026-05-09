from __future__ import annotations

from typing import Any

from common.config_utils import cfg_get


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def replay_mix_recent_label(cfg) -> str:
    mix_recent = _as_bool(cfg_get(cfg, "alg.replay.mix_recent", default=False))
    return "mr1" if mix_recent else "mr0"
