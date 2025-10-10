from __future__ import annotations

import logging
import sys
import os
import warnings
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOGGER_NAME = "painter"
_CONFIGURED = False
_STREAM_H: Optional[logging.Handler] = None
_FILE_H: Optional[logging.Handler] = None


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[36m",     # cyan
        logging.INFO: "\x1b[32m",      # green
        logging.WARNING: "\x1b[33m",   # yellow
        logging.ERROR: "\x1b[31m",     # red
        logging.CRITICAL: "\x1b[35m",  # magenta
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        levelname = record.levelname
        if color:
            record.levelname = f"{color}{levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = levelname  # restore


def _to_level(level: str) -> int:
    lvl = (level or "INFO").upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(lvl, logging.INFO)


def _install_exception_hook(logger: logging.Logger) -> None:
    def _hook(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _hook


def configure_logging(
    level: str = "INFO",
    *,
    file_path: str = "../logs/painter.log",
    file_max_bytes: int = 2 * 1024 * 1024,
    file_backup_count: int = 5,
) -> None:
    """
    Configure 'painter' logger:
      - colored console with selected level (default INFO)
      - rotating file handler with DEBUG level
      - capture warnings and uncaught exceptions
    Idempotent: repeated calls only adjust levels.
    """
    global _CONFIGURED, _STREAM_H, _FILE_H

    logger = logging.getLogger(_LOGGER_NAME)

    if not _CONFIGURED:
        logger.setLevel(logging.DEBUG)

        _STREAM_H = logging.StreamHandler(stream=sys.stdout)
        _STREAM_H.setLevel(_to_level(level))
        _STREAM_H.setFormatter(
            _ColorFormatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(_STREAM_H)

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except Exception:
            pass
        _FILE_H = RotatingFileHandler(
            file_path,
            maxBytes=int(file_max_bytes),
            backupCount=int(file_backup_count),
            encoding="utf-8",
        )
        _FILE_H.setLevel(logging.DEBUG)
        _FILE_H.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(_FILE_H)

        logger.propagate = False
        warnings.simplefilter("default")
        logging.captureWarnings(True)
        _install_exception_hook(logger)

        _CONFIGURED = True
    else:
        if _STREAM_H is not None:
            _STREAM_H.setLevel(_to_level(level))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    full = _LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full)
