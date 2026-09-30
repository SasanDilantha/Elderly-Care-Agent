"""Thin command-line delivery layer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from elderly_care_agent.application.services import AnnotationValidationService
from elderly_care_agent.config import ApplicationSettings
from elderly_care_agent.domain.exceptions import AnnotationFormatError, DatasetPipelineError
from elderly_care_agent.infrastructure.json_annotation_repository import (
    JsonAnnotationRepository,
)
from elderly_care_agent.infrastructure.logging_config import LoggingConfigurator
from elderly_care_agent.pipeline.dataset_pipeline import DatasetPreparationPipeline

logger = logging.getLogger(__name__)


class CliApplication:
    """Parses commands and delegates work to application services."""

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        validation_service: AnnotationValidationService | None = None,
        output: TextIO | None = None,
        logging_configurator: LoggingConfigurator | None = None,
    ) -> None:
        self._settings = settings or ApplicationSettings()
        self._validation_service = validation_service or AnnotationValidationService(
            JsonAnnotationRepository()
        )
        self._output = output or sys.stdout
        self._logging_configurator = logging_configurator

    def run(self, arguments: Sequence[str] | None = None) -> int:
        parser = self._build_parser()
        namespace = parser.parse_args(arguments)
        if self._logging_configurator is not None:
            log_file = None if namespace.console_only else namespace.log_file
            self._logging_configurator.configure(namespace.log_level, log_file)
        logger.info("Command started | command=%s", namespace.command)

        if namespace.command == "show-config":
            self._write_json(self._settings.to_dict())
            return 0

        if namespace.command == "validate-annotations":
            try:
                report = self._validation_service.validate(namespace.path)
            except AnnotationFormatError as error:
                logger.error("Annotation validation failed | error=%s", error)
                self._write_json({"valid": False, "error": str(error)})
                return 1
            self._write_json({"valid": True, **asdict(report)})
            return 0

        if namespace.command == "prepare-dataset":
            try:
                report = DatasetPreparationPipeline.from_config(namespace.config).run(
                    force=namespace.force
                )
            except (DatasetPipelineError, OSError) as error:
                logger.error("Dataset preparation failed | error=%s", error)
                self._write_json({"status": "failed", "error": str(error)})
                return 1
            self._write_json(report.to_dict())
            return 0

        parser.error("a command is required")
        return 2

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="elderly-care-agent")
        parser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default=os.environ.get("ELDERLY_CARE_LOG_LEVEL", "INFO").upper(),
            help="terminal and file logging detail",
        )
        parser.add_argument(
            "--log-file",
            type=Path,
            default=Path(
                os.environ.get(
                    "ELDERLY_CARE_LOG_FILE",
                    "logs/elderly-care-agent.log",
                )
            ),
            help="rotating application log path",
        )
        parser.add_argument(
            "--console-only",
            action="store_true",
            help="show logs in the terminal without writing a log file",
        )
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("show-config", help="print the active settings")
        validate_parser = subparsers.add_parser(
            "validate-annotations",
            help="validate one ground-truth JSON file",
        )
        validate_parser.add_argument("path", type=Path)
        prepare_parser = subparsers.add_parser(
            "prepare-dataset",
            help="download, verify, select, and validate the configured dataset",
        )
        prepare_parser.add_argument(
            "--config",
            type=Path,
            default=Path("config/dataset.yml"),
            help="dataset YAML configuration path",
        )
        prepare_parser.add_argument(
            "--force",
            action="store_true",
            help="replace a generated staged file only when its checksum differs",
        )
        return parser

    def _write_json(self, payload: object) -> None:
        json.dump(payload, self._output, indent=2)
        self._output.write("\n")


def main() -> None:
    """Minimal composition root used by the installed console command."""

    raise SystemExit(CliApplication(logging_configurator=LoggingConfigurator()).run())
