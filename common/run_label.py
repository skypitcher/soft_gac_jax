def _resolve_alg_class(alg_name: str):
    alg_name = str(alg_name)
    if alg_name == "soft_gac":
        from algorithms.soft_gac import SoftGAC

        return SoftGAC
    if alg_name == "sac":
        from algorithms.sac import SAC

        return SAC
    if alg_name == "qsm":
        from algorithms.qsm import QSM

        return QSM
    if alg_name == "flowrl":
        from algorithms.flowrl import FlowRL

        return FlowRL
    if alg_name == "flac":
        from algorithms.flac import FLAC

        return FLAC
    if alg_name == "dime":
        from algorithms.dime import DIME

        return DIME
    if alg_name == "qvpo":
        from algorithms.qvpo import QVPO

        return QVPO
    raise ValueError(f"Unsupported algorithm for automatic labels: {alg_name}")


def _extra_label_suffix(cfg) -> str:
    extra_label = cfg.get("extra_label")
    if extra_label is None or extra_label is False:
        return ""
    suffix = str(extra_label).strip()
    if not suffix:
        return ""
    suffix = suffix.strip("_")
    if not suffix:
        return ""
    return "_" + suffix.replace("/", "_").replace("-", "_").replace(" ", "_")


def resolved_run_label(cfg) -> str:
    if cfg.get("run_label"):
        base_label = str(cfg.run_label)
    else:
        alg_cls = _resolve_alg_class(cfg.alg.name)
        if not hasattr(alg_cls, "default_run_label"):
            raise AttributeError(f"{alg_cls.__name__} does not define default_run_label(cfg)")
        base_label = str(alg_cls.default_run_label(cfg))
    return base_label + _extra_label_suffix(cfg)
