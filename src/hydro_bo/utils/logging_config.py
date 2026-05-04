"""
Structured logging configuration.

structlog and stdlib `logging` are wired together via
`structlog.stdlib.ProcessorFormatter` so a single configuration call
produces:

  - a clean, ANSI-free log file (always),
  - an optional colorised stderr stream (when tee_stderr=True).

stdlib loggers (gurobi, pyomo, jax, ...) flow through the same pipeline,
so their output picks up the structlog format and shows up in the file
in the same shape as our own structlog events.
"""

import logging
import sys
from pathlib import Path

import structlog


_file_handler: logging.FileHandler | None = None
_stderr_handler: logging.StreamHandler | None = None


def _shared_processors():
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]


def configure_logging(level=logging.INFO, log_file: str | Path | None = None,
                      tee_stderr: bool = True, package_level: int | None = None):
    """Configure structlog + stdlib logging.

    Args:
        level: External / root logger level — controls third-party
            libraries (gurobi, pyomo, jax, ray, ...) that route through
            stdlib but inherit from root. Default logging.INFO.
        log_file: Path for the log file (default: ./debug_mpc/run.log).
        tee_stderr: Mirror output to stderr in addition to the file
            (default: True). Set False for per-worker logs so Ray's
            captured stderr stays quiet.
        package_level: Override level for the `hydro_bo` namespace and
            our structlog calls. Defaults to `level` when None — set it
            independently to verbose-up our own modules (e.g. DEBUG for
            BO internals) without unleashing third-party log spam.
    """
    global _file_handler, _stderr_handler

    if package_level is None:
        package_level = level

    log_path = Path(log_file) if log_file is not None else Path("./debug_mpc/run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    shared = _shared_processors()

    structlog.configure(
        processors=shared + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(package_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )
    stderr_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )

    root_logger = logging.getLogger()
    if _file_handler is not None:
        root_logger.removeHandler(_file_handler)
        try:
            _file_handler.close()
        except Exception:
            pass
        _file_handler = None
    if _stderr_handler is not None:
        root_logger.removeHandler(_stderr_handler)
        _stderr_handler = None
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Handlers run at the most permissive of (root, package) so neither
    # logger's own filtering is undermined by a stricter handler gate.
    handler_level = min(level, package_level)

    _file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _file_handler.setFormatter(file_formatter)
    _file_handler.setLevel(handler_level)
    root_logger.addHandler(_file_handler)

    if tee_stderr:
        _stderr_handler = logging.StreamHandler(sys.stderr)
        _stderr_handler.setFormatter(stderr_formatter)
        _stderr_handler.setLevel(handler_level)
        root_logger.addHandler(_stderr_handler)

    root_logger.setLevel(level)
    logging.getLogger("hydro_bo").setLevel(package_level)


def get_logger(name: str):
    """Get a configured structlog logger."""
    return structlog.get_logger(name)
