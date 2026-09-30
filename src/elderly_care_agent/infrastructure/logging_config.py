"""Central application logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO


class LoggingConfigurator:
    """Configures consistent console and rotating-file observability."""

    _FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, console: TextIO | None = None) -> None:
        self._console = console or sys.stderr

    def configure(self, level: str, log_file: Path | None) -> None:
        """Replace root handlers with deterministic application handlers."""

        numeric_level = logging.getLevelNamesMapping().get(level.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"unsupported log level: {level}")

        formatter = logging.Formatter(self._FORMAT, datefmt=self._DATE_FORMAT)
        console_handler = logging.StreamHandler(self._console)
        console_handler.setFormatter(formatter)
        handlers: list[logging.Handler] = [console_handler]

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        logging.basicConfig(
            level=numeric_level,
            handlers=handlers,
            force=True,
        )
        logging.captureWarnings(True)
        for noisy_logger in ("matplotlib", "PIL"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        destination = str(log_file) if log_file is not None else "disabled"
        logging.getLogger(__name__).info(
            "Logging initialized | level=%s | file=%s",
            level.upper(),
            destination,
        )
