"""Thin command-line delivery layer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, TextIO
import sys

from elderly_care_agent.application.services import AnnotationValidationService
from elderly_care_agent.config import ApplicationSettings
from elderly_care_agent.domain.exceptions import AnnotationFormatError
from elderly_care_agent.infrastructure.json_annotation_repository import (
    JsonAnnotationRepository,
)


class CliApplication:
    """Parses commands and delegates work to application services."""

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        validation_service: AnnotationValidationService | None = None,
        output: TextIO | None = None,
    ) -> None:
        self._settings = settings or ApplicationSettings()
        self._validation_service = validation_service or AnnotationValidationService(
            JsonAnnotationRepository()
        )
        self._output = output or sys.stdout

    def run(self, arguments: Sequence[str] | None = None) -> int:
        parser = self._build_parser()
        namespace = parser.parse_args(arguments)

        if namespace.command == "show-config":
            self._write_json(self._settings.to_dict())
            return 0

        if namespace.command == "validate-annotations":
            try:
                report = self._validation_service.validate(namespace.path)
            except AnnotationFormatError as error:
                self._write_json({"valid": False, "error": str(error)})
                return 1
            self._write_json({"valid": True, **asdict(report)})
            return 0

        parser.error("a command is required")
        return 2

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="elderly-care-agent")
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("show-config", help="print the active settings")
        validate_parser = subparsers.add_parser(
            "validate-annotations",
            help="validate one ground-truth JSON file",
        )
        validate_parser.add_argument("path", type=Path)
        return parser

    def _write_json(self, payload: object) -> None:
        json.dump(payload, self._output, indent=2)
        self._output.write("\n")


def main() -> None:
    """Minimal composition root used by the installed console command."""

    raise SystemExit(CliApplication().run())
