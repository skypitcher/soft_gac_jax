import logging
import sys
from pathlib import Path

LOGGER_NAME = "rl"

COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        orig = record.levelname
        color = COLORS.get(orig, "")
        reset = COLORS["RESET"]
        record.levelname = f"{color}{orig:<7}{reset}"
        result = super().format(record)
        record.levelname = orig
        return result


class _StreamToLogger:
    """File-like stream wrapper that forwards writes into a logger."""

    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self._buffer = ""

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")

    def isatty(self):
        return False

    def fileno(self):
        return self.original_stream.fileno()

    def write(self, message):
        if not message:
            return 0

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self._infer_level(line), line)
        return len(message)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self._infer_level(self._buffer.strip()), self._buffer.strip())
        self._buffer = ""

    def _infer_level(self, line):
        if self.level != logging.ERROR:
            return self.level

        text = line.strip()
        lower = text.lower()

        if text.startswith("wandb:"):
            if any(tok in lower for tok in ("error", "failed", "exception", "traceback")):
                return logging.ERROR
            if any(tok in lower for tok in ("warning", "warn")):
                return logging.WARNING
            return logging.INFO

        if "traceback" in lower:
            return logging.ERROR
        if lower.startswith("error") or " error " in lower:
            return logging.ERROR
        if lower.startswith("warning") or " warning " in lower:
            return logging.WARNING

        return logging.WARNING


def _has_file_handler(logger, log_file):
    target = str(Path(log_file).resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == Path(target):
                    return True
            except OSError:
                if handler.baseFilename == target:
                    return True
    return False


def _resolve_console_stream(stream):
    while isinstance(stream, _StreamToLogger):
        stream = stream.original_stream
    return stream


def _build_console_formatter(stream):
    use_color = bool(getattr(stream, "isatty", lambda: False)())
    if use_color:
        return _ColorFormatter(
            fmt=f"{COLORS['DIM']}%(asctime)s{COLORS['RESET']} │ %(levelname)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _configure_console_handler(logger):
    stream = _resolve_console_stream(sys.stdout)
    console = None
    for handler in logger.handlers:
        if getattr(handler, "_rl_console_handler", False):
            console = handler
            break

    if console is None:
        console = logging.StreamHandler(stream)
        console._rl_console_handler = True
        logger.addHandler(console)
    else:
        console.setStream(stream)

    console.setLevel(logging.DEBUG)
    console.setFormatter(_build_console_formatter(stream))


def _redirect_std_streams(logger):
    if not isinstance(sys.stdout, _StreamToLogger):
        sys.stdout = _StreamToLogger(logger, logging.INFO, sys.stdout)
    if not isinstance(sys.stderr, _StreamToLogger):
        sys.stderr = _StreamToLogger(logger, logging.ERROR, sys.stderr)


def setup_logger(name=LOGGER_NAME, log_file=None, capture_stdio=False):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    _configure_console_handler(logger)

    if log_file and not _has_file_handler(logger, log_file):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    if capture_stdio:
        _redirect_std_streams(logger)

    return logger


def compute_eval_stats(rewards):
    import numpy as np

    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        return {
            "trimmed_rewards": rewards,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_total": 0,
            "n_trimmed_each_side": 0,
            "n_used": 0,
        }

    xs = np.sort(rewards)
    return {
        "trimmed_rewards": xs,
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
        "n_total": int(xs.size),
        "n_trimmed_each_side": 0,
        "n_used": int(xs.size),
    }


def log_eval(logger, config, rewards, elapsed, total_steps):
    del total_steps
    stats = compute_eval_stats(rewards)
    sep = "═" * 60
    env_label = f"{config.domain}-{config.task}" if config.domain else config.task
    logger.info(sep)
    logger.info(
        "Eval │ %s │ %d eps │ mean: %.5f │ std: %.5f │ min: %.5f │ max: %.5f │ %.5fs",
        env_label,
        stats["n_total"],
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["max"],
        elapsed,
    )
    logger.info(sep)


def log_episode(logger, episode, total_steps, ep_steps, reward, elapsed, total_elapsed):
    logger.info(
        "Ep %-5d │ %10.5fk steps │ %4d ep_steps │ reward: %10.5f │ %8.5fs │ total: %s",
        episode,
        total_steps / 1e3,
        ep_steps,
        reward,
        elapsed,
        _fmt_duration(total_elapsed),
    )


def _fmt_duration(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_metric_value(value):
    try:
        value = float(value)
    except Exception:
        return str(value)

    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    if abs_value >= 100:
        return f"{value:.3f}"
    if abs_value >= 1:
        return f"{value:.4f}"
    return f"{value:.5f}"


def log_train_metrics(logger, metrics):
    if not metrics:
        return
    parts = [f"{key}={_fmt_metric_value(value)}" for key, value in metrics.items()]
    logger.info("Train   │ %s", " │ ".join(parts))
