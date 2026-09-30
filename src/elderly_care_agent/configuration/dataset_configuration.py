"""Strict YAML-to-entity configuration adapter for dataset preparation."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from elderly_care_agent.domain.dataset import DatasetIngestionConfig, DatasetSource
from elderly_care_agent.domain.exceptions import DatasetConfigurationError

logger = logging.getLogger(__name__)


class DatasetConfigurationManager:
    """Loads one dataset config without leaking YAML dictionaries downstream."""

    def __init__(
        self,
        config_path: Path = Path("config/dataset.yml"),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._config_path = config_path
        self._environment = environment if environment is not None else os.environ

    def load(self) -> DatasetIngestionConfig:
        """Parse, validate, and resolve all configured paths."""

        logger.info("Loading dataset configuration | path=%s", self._config_path)
        try:
            with self._config_path.open(encoding="utf-8") as config_file:
                raw = yaml.safe_load(config_file)
        except (OSError, yaml.YAMLError) as error:
            raise DatasetConfigurationError(
                f"cannot read dataset configuration {self._config_path}: {error}"
            ) from error

        root = self._mapping(raw, "root")
        dataset = self._mapping(root.get("dataset"), "dataset")
        source_raw = self._mapping(dataset.get("source"), "dataset.source")
        ingestion = self._mapping(dataset.get("ingestion"), "dataset.ingestion")

        project_root = self._config_path.resolve().parent.parent
        cache_env = self._text(ingestion, "cache_dir_env")
        configured_cache_dir = self._text(ingestion, "cache_dir")
        cache_dir = self._project_path(
            project_root,
            self._environment.get(cache_env, configured_cache_dir),
        ).expanduser()

        source = DatasetSource(
            owner=self._text(source_raw, "owner"),
            canonical_repository=self._url(source_raw, "canonical_repository"),
            mirror_repository=self._url(source_raw, "mirror_repository"),
            mirror_commit=self._commit(source_raw, "mirror_commit"),
            archive_url=self._url(source_raw, "archive_url"),
            archive_checksum=self._checksum(source_raw, "archive_checksum"),
            dataset_doi=self._text(source_raw, "dataset_doi"),
            article_doi=self._text(source_raw, "article_doi"),
            dataset_license=self._text(source_raw, "dataset_license"),
            repository_license=self._text(source_raw, "repository_license"),
        )

        expected_subjects_raw = ingestion.get("expected_subjects")
        if not isinstance(expected_subjects_raw, list) or not expected_subjects_raw:
            raise DatasetConfigurationError("dataset.ingestion.expected_subjects must be a list")
        expected_subjects = tuple(str(value).strip() for value in expected_subjects_raw)
        if any(not subject for subject in expected_subjects):
            raise DatasetConfigurationError("expected subjects must not be empty")

        config = DatasetIngestionConfig(
            dataset_name=self._text(dataset, "name"),
            dataset_version=self._text(dataset, "version"),
            source=source,
            archive_path=cache_dir / self._text(ingestion, "archive_name"),
            source_dir=cache_dir / self._text(ingestion, "source_directory_name"),
            selection_manifest=self._project_path(
                project_root, self._text(ingestion, "selection_manifest")
            ),
            output_dir=self._project_path(project_root, self._text(ingestion, "output_dir")),
            report_path=self._project_path(project_root, self._text(ingestion, "report_path")),
            inventory_path=self._project_path(
                project_root, self._text(ingestion, "inventory_path")
            ),
            expected_video_count=self._positive_int(ingestion, "expected_video_count"),
            expected_metadata_csv_count=self._positive_int(
                ingestion, "expected_metadata_csv_count"
            ),
            expected_subjects=expected_subjects,
        )
        logger.info(
            "Dataset configuration ready | cache=%s | raw=%s | manifest=%s",
            config.archive_path.parent,
            config.output_dir,
            config.selection_manifest,
        )
        return config

    @staticmethod
    def _mapping(value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise DatasetConfigurationError(f"{name} must be a mapping")
        return value

    @staticmethod
    def _text(mapping: Mapping[str, Any], key: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DatasetConfigurationError(f"{key} must be a non-empty string")
        return value.strip()

    @classmethod
    def _url(cls, mapping: Mapping[str, Any], key: str) -> str:
        value = cls._text(mapping, key)
        if not value.startswith("https://"):
            raise DatasetConfigurationError(f"{key} must use HTTPS")
        return value

    @classmethod
    def _commit(cls, mapping: Mapping[str, Any], key: str) -> str:
        value = cls._text(mapping, key).lower()
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise DatasetConfigurationError(f"{key} must be a full 40-character Git SHA")
        return value

    @classmethod
    def _checksum(cls, mapping: Mapping[str, Any], key: str) -> str:
        value = cls._text(mapping, key).lower()
        parts = value.split(":", maxsplit=1)
        expected_lengths = {"md5": 32, "sha256": 64}
        if (
            len(parts) != 2
            or parts[0] not in expected_lengths
            or len(parts[1]) != expected_lengths[parts[0]]
            or any(character not in "0123456789abcdef" for character in parts[1])
        ):
            raise DatasetConfigurationError(f"{key} must be md5:<hex> or sha256:<hex>")
        return value

    @staticmethod
    def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
        value = mapping.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DatasetConfigurationError(f"{key} must be a positive integer")
        return value

    @staticmethod
    def _project_path(project_root: Path, configured_path: str) -> Path:
        path = Path(configured_path)
        return path if path.is_absolute() else project_root / path
