def canonical_base_distribution(value: str | None) -> str:
    raw = "logistic" if value is None else str(value).strip().lower()
    aliases = {
        "log": "logistic",
        "logistic": "logistic",
        "normal": "normal",
        "gauss": "normal",
        "gaussian": "normal",
        "n01": "normal",
        "standard_normal": "normal",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported SoftGAC base distribution '{value}'. Supported aliases: {supported}") from exc


def base_distribution_label(value: str | None) -> str:
    base = canonical_base_distribution(value)
    if base == "logistic":
        return "p0log"
    if base == "normal":
        return "p0gauss"
    raise AssertionError(f"Unhandled SoftGAC base distribution: {base}")
