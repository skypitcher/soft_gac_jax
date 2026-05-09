from __future__ import annotations

from collections.abc import Mapping


_MISSING = object()


def _select(config, path: str):
    current = config
    for part in path.split("."):
        if current is None:
            return _MISSING
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        try:
            current = getattr(current, part)
        except Exception:
            return _MISSING
    return current


def cfg_get(config, new_path: str, old_path: str | None = None, default=_MISSING):
    value = _select(config, new_path)
    if value is not _MISSING:
        return value
    if old_path is not None:
        value = _select(config, old_path)
        if value is not _MISSING:
            return value
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing config key: {new_path}" + (f" (fallback {old_path})" if old_path else ""))
