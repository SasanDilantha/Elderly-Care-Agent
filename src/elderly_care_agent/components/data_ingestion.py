"""Idempotent, integrity-checked GMDCSA-24 preparation component."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from elderly_care_agent.application.dataset_ports import (
    DatasetDownloader,
    DatasetExtractor,
    DatasetSelectionRepository,
    VideoProbe,
)
from elderly_care_agent.domain.dataset import (
    DatasetIngestionConfig,
    DatasetPreparationReport,
    DatasetSelection,
    VideoInventoryItem,
)
from elderly_care_agent.domain.exceptions import DatasetIntegrityError
from elderly_care_agent.infrastructure.dataset_io import FileHasher

logger = logging.getLogger(__name__)


class DataIngestion:
    """Acquires the official data and stages only assignment-relevant videos."""

    def __init__(
        self,
        config: DatasetIngestionConfig,
        downloader: DatasetDownloader,
        extractor: DatasetExtractor,
        selections: DatasetSelectionRepository,
        video_probe: VideoProbe,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._downloader = downloader
        self._extractor = extractor
        self._selections = selections
        self._video_probe = video_probe
        self._clock = clock or (lambda: datetime.now(UTC))

    def run(self, force: bool = False) -> DatasetPreparationReport:
        """Run acquisition, selection, validation, inventory, and reporting."""

        logger.info(
            "Dataset preparation started | dataset=%s | version=%s | output=%s | force=%s",
            self._config.dataset_name,
            self._config.dataset_version,
            self._config.output_dir,
            force,
        )
        selected = self._selections.load(self._config.selection_manifest)
        unexpected_subjects = sorted(
            {item.subject for item in selected} - set(self._config.expected_subjects)
        )
        if unexpected_subjects:
            raise DatasetIntegrityError(
                f"manifest contains unexpected subjects: {', '.join(unexpected_subjects)}"
            )

        if not force and self._prepared_dataset_is_available(selected):
            logger.info(
                "Prepared dataset already available; acquisition skipped | output=%s | videos=%d",
                self._config.output_dir,
                len(selected),
            )
            return self._already_prepared_report(selected)

        logger.info("Prepared dataset missing or refresh requested; acquisition will run")
        source_mode = self._ensure_source_available()

        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._stage_source_documents(force)
        self._stage_metadata_files(selected, force)

        copied_count = 0
        reused_count = 0
        inventory: list[VideoInventoryItem] = []
        total_selected = len(selected)
        for index, item in enumerate(selected, start=1):
            source_path = self._config.source_dir / item.source_relative_path
            if not source_path.is_file():
                raise DatasetIntegrityError(f"selected source video is missing: {source_path}")
            destination_path = self._config.output_dir / item.destination_relative_path
            source_sha256 = FileHasher.digest(source_path)
            copied = self._copy_verified_file(
                source_path,
                destination_path,
                source_sha256,
                force,
            )
            copied_count += int(copied)
            reused_count += int(not copied)
            logger.debug(
                "Staged video | completed=%d/%d | action=%s | source=%s | destination=%s",
                index,
                total_selected,
                "copied" if copied else "reused",
                source_path,
                destination_path,
            )

            probe = self._video_probe.inspect(destination_path)
            inventory.append(
                VideoInventoryItem(
                    split=item.split,
                    subject=item.subject,
                    category=item.category,
                    video=item.video,
                    relative_path=destination_path.relative_to(self._config.output_dir).as_posix(),
                    sha256=source_sha256,
                    size_bytes=destination_path.stat().st_size,
                    frame_count=probe.frame_count,
                    fps=probe.fps,
                    duration_sec=probe.duration_sec,
                )
            )
            if index % 5 == 0 or index == total_selected:
                logger.info(
                    "Video preparation progress | completed=%d/%d | copied=%d | reused=%d",
                    index,
                    total_selected,
                    copied_count,
                    reused_count,
                )

        self._write_inventory(inventory)
        split_counts = dict(sorted(Counter(item.split for item in selected).items()))
        report = DatasetPreparationReport(
            status="ready",
            dataset_name=self._config.dataset_name,
            dataset_version=self._config.dataset_version,
            source_owner=self._config.source.owner,
            source_doi=self._config.source.dataset_doi,
            source_repository=self._config.source.canonical_repository,
            mirror_repository=self._config.source.mirror_repository,
            mirror_commit=self._config.source.mirror_commit,
            dataset_license=self._config.source.dataset_license,
            source_mode=source_mode,
            manifest_sha256=FileHasher.digest(self._config.selection_manifest),
            selected_video_count=len(selected),
            copied_video_count=copied_count,
            reused_video_count=reused_count,
            total_size_bytes=sum(item.size_bytes for item in inventory),
            split_counts=split_counts,
            report_path=str(self._config.report_path),
            inventory_path=str(self._config.inventory_path),
            completed_at_utc=self._clock().astimezone(UTC).isoformat(),
        )
        self._write_json(self._config.report_path, report.to_dict())
        logger.info(
            "Dataset preparation completed | copied=%d | reused=%d | bytes=%d | report=%s",
            copied_count,
            reused_count,
            report.total_size_bytes,
            self._config.report_path,
        )
        return report

    def _prepared_dataset_is_available(
        self,
        selected: tuple[DatasetSelection, ...],
    ) -> bool:
        """Return true only when every required staged file is already present."""

        required_paths = {
            self._config.output_dir / "LICENSE",
            self._config.output_dir / "SOURCE_README.md",
        }
        required_paths.update(
            self._config.output_dir / item.destination_relative_path for item in selected
        )
        required_paths.update(
            self._config.output_dir / item.split / item.subject / f"{item.category}.csv"
            for item in selected
        )
        return bool(required_paths) and all(path.is_file() for path in required_paths)

    def _already_prepared_report(
        self,
        selected: tuple[DatasetSelection, ...],
    ) -> DatasetPreparationReport:
        """Describe a no-op run without requiring the archive or extracted source."""

        split_counts = dict(sorted(Counter(item.split for item in selected).items()))
        total_size_bytes = sum(
            (self._config.output_dir / item.destination_relative_path).stat().st_size
            for item in selected
        )
        return DatasetPreparationReport(
            status="already_ready",
            dataset_name=self._config.dataset_name,
            dataset_version=self._config.dataset_version,
            source_owner=self._config.source.owner,
            source_doi=self._config.source.dataset_doi,
            source_repository=self._config.source.canonical_repository,
            mirror_repository=self._config.source.mirror_repository,
            mirror_commit=self._config.source.mirror_commit,
            dataset_license=self._config.source.dataset_license,
            source_mode="prepared_output",
            manifest_sha256=FileHasher.digest(self._config.selection_manifest),
            selected_video_count=len(selected),
            copied_video_count=0,
            reused_video_count=len(selected),
            total_size_bytes=total_size_bytes,
            split_counts=split_counts,
            report_path=str(self._config.report_path),
            inventory_path=str(self._config.inventory_path),
            completed_at_utc=self._clock().astimezone(UTC).isoformat(),
        )

    def _ensure_source_available(self) -> str:
        if self._config.source_dir.exists():
            logger.info("Using extracted dataset source | path=%s", self._config.source_dir)
            self._validate_source_layout(self._config.source_dir)
            return "existing_source_directory"

        downloaded = False
        if self._config.archive_path.is_file():
            logger.info("Using cached dataset archive | path=%s", self._config.archive_path)
            try:
                FileHasher.verify(
                    self._config.archive_path,
                    self._config.source.archive_checksum,
                )
            except DatasetIntegrityError as error:
                logger.warning(
                    "Cached archive is invalid and will be replaced | path=%s | error=%s",
                    self._config.archive_path,
                    error,
                )
                self._config.archive_path.unlink()

        if not self._config.archive_path.is_file():
            logger.info(
                "Dataset archive unavailable; download required | source=%s",
                self._config.source.archive_url,
            )
            self._downloader.download(
                self._config.source.archive_url,
                self._config.archive_path,
            )
            downloaded = True
            try:
                FileHasher.verify(
                    self._config.archive_path,
                    self._config.source.archive_checksum,
                )
            except DatasetIntegrityError:
                logger.error(
                    "Downloaded archive failed checksum and will be removed | path=%s",
                    self._config.archive_path,
                )
                self._config.archive_path.unlink(missing_ok=True)
                raise

        logger.info(
            "Preparing extracted source directory | archive=%s | destination=%s",
            self._config.archive_path,
            self._config.source_dir,
        )
        self._config.source_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=self._config.source_dir.parent,
            prefix=".gmdcsa24-extract-",
        ) as temporary_directory:
            extracted_dir = Path(temporary_directory) / "dataset"
            self._extractor.extract(self._config.archive_path, extracted_dir)
            self._validate_source_layout(extracted_dir)
            os.replace(extracted_dir, self._config.source_dir)
        logger.info("Extracted source published | path=%s", self._config.source_dir)
        return "downloaded_archive" if downloaded else "verified_archive"

    def _validate_source_layout(self, source_dir: Path) -> None:
        logger.info("Validating dataset source layout | path=%s", source_dir)
        video_count = sum(1 for _ in source_dir.rglob("*.mp4"))
        metadata_count = sum(1 for _ in source_dir.rglob("*.csv"))
        actual_subjects = {
            path.name
            for path in source_dir.iterdir()
            if path.is_dir() and path.name.startswith("Subject ")
        }
        expected_subjects = set(self._config.expected_subjects)
        errors: list[str] = []
        if video_count != self._config.expected_video_count:
            errors.append(
                f"expected {self._config.expected_video_count} videos, found {video_count}"
            )
        if metadata_count != self._config.expected_metadata_csv_count:
            errors.append(
                f"expected {self._config.expected_metadata_csv_count} CSV files, "
                f"found {metadata_count}"
            )
        if actual_subjects != expected_subjects:
            errors.append(
                f"expected subjects {sorted(expected_subjects)}, found {sorted(actual_subjects)}"
            )
        for required_file in ("LICENSE", "README.md"):
            if not (source_dir / required_file).is_file():
                errors.append(f"missing {required_file}")
        if errors:
            raise DatasetIntegrityError(
                f"invalid GMDCSA-24 source directory {source_dir}: {'; '.join(errors)}"
            )
        logger.info(
            "Dataset source layout valid | videos=%d | metadata_csv=%d | subjects=%d",
            video_count,
            metadata_count,
            len(actual_subjects),
        )

    def _stage_source_documents(self, force: bool) -> None:
        logger.info("Staging source attribution documents")
        documents = {
            self._config.source_dir / "LICENSE": self._config.output_dir / "LICENSE",
            self._config.source_dir / "README.md": self._config.output_dir / "SOURCE_README.md",
        }
        for source_path, destination_path in documents.items():
            self._copy_verified_file(
                source_path,
                destination_path,
                FileHasher.digest(source_path),
                force,
            )

    def _stage_metadata_files(
        self,
        selected: tuple[DatasetSelection, ...],
        force: bool,
    ) -> None:
        metadata_identities = {(item.split, item.subject, item.category) for item in selected}
        logger.info("Staging source metadata | files=%d", len(metadata_identities))
        for split, subject, category in sorted(metadata_identities):
            source_path = self._config.source_dir / subject / f"{category}.csv"
            destination_path = self._config.output_dir / split / subject / f"{category}.csv"
            if not source_path.is_file():
                raise DatasetIntegrityError(f"source metadata is missing: {source_path}")
            self._copy_verified_file(
                source_path,
                destination_path,
                FileHasher.digest(source_path),
                force,
            )

    @staticmethod
    def _copy_verified_file(
        source_path: Path,
        destination_path: Path,
        source_sha256: str,
        force: bool,
    ) -> bool:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.is_file():
            destination_sha256 = FileHasher.digest(destination_path)
            if destination_sha256 == source_sha256:
                return False
            if not force:
                raise DatasetIntegrityError(
                    f"staged file differs from source: {destination_path}; rerun with --force "
                    "to replace only this generated file"
                )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination_path.parent,
                prefix=f".{destination_path.name}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            shutil.copy2(source_path, temporary_path)
            if FileHasher.digest(temporary_path) != source_sha256:
                raise DatasetIntegrityError(f"copy verification failed: {destination_path}")
            os.replace(temporary_path, destination_path)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_inventory(self, inventory: list[VideoInventoryItem]) -> None:
        logger.info(
            "Writing verified video inventory | path=%s | rows=%d",
            self._config.inventory_path,
            len(inventory),
        )
        self._config.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(inventory[0]).keys())
        temporary_path = self._config.inventory_path.with_suffix(".csv.part")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as inventory_file:
                writer = csv.DictWriter(inventory_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(asdict(item) for item in inventory)
            os.replace(temporary_path, self._config.inventory_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.part")
        try:
            with temporary_path.open("w", encoding="utf-8") as report_file:
                json.dump(payload, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
