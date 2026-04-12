"""
Structured logging configuration for MPC controller
"""

import logging
import sys
from pathlib import Path

import structlog


_log_file_handle = None


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for stream in self._streams:
            stream.write(text)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def configure_logging(level=logging.INFO, log_file: str | Path | None = None):
    """Configure structlog with appropriate processors and formatting.

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional path for logfile (default: ./debug_mpc/run.log)
    """
    global _log_file_handle

    log_path = Path(log_file) if log_file is not None else Path("./debug_mpc/run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except Exception:
            pass

    # Overwrite logfile on each new run.
    _log_file_handle = open(log_path, "w", buffering=1)
    tee_stream = _TeeStream(sys.stderr, _log_file_handle)

    # Reset root handlers so repeated configure calls don't duplicate logs.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set up Python logging (including gurobipy logger) to both stderr + file.
    logging.basicConfig(level=level, stream=tee_stream)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=tee_stream),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str):
    """Get a configured structlog logger."""
    return structlog.get_logger(name)
