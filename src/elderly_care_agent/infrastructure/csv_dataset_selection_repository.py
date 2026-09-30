"""CSV adapter for the version-controlled GMDCSA-24 subset manifest."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from elderly_care_agent.application.dataset_ports import DatasetSelectionRepository
from elderly_care_agent.domain.dataset import DatasetSelection
from elderly_care_agent.domain.exceptions import DatasetIntegrityError

logger = logging.getLogger(__name__)


class CsvDatasetSelectionRepository(DatasetSelectionRepository):
    """Converts manifest rows into validated immutable selections."""

    _REQUIRED_FIELDS = {
        "split",
        "subject",
        "category",
        "video",
        "coverage",
        "event_focus",
    }
    _ALLOWED_SPLITS = {"development", "evaluation"}
    _ALLOWED_CATEGORIES = {"ADL", "Fall"}

    def load(self, manifest_path: Path) -> tuple[DatasetSelection, ...]:
        logger.info("Loading dataset selection manifest | path=%s", manifest_path)
        try:
            with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
                reader = csv.DictReader(manifest_file)
                if not self._REQUIRED_FIELDS.issubset(reader.fieldnames or []):
                    missing = sorted(self._REQUIRED_FIELDS - set(reader.fieldnames or []))
                    raise DatasetIntegrityError(
                        f"selection manifest is missing columns: {', '.join(missing)}"
                    )
                selections = tuple(
                    self._selection_from_row(row, row_number)
                    for row_number, row in enumerate(reader, start=2)
                )
        except OSError as error:
            raise DatasetIntegrityError(f"cannot read selection manifest: {error}") from error

        if not selections:
            raise DatasetIntegrityError("selection manifest must contain at least one video")
        self._validate_unique_sources(selections)
        self._validate_subject_isolation(selections)
        split_counts: dict[str, int] = {}
        for selection in selections:
            split_counts[selection.split] = split_counts.get(selection.split, 0) + 1
        logger.info(
            "Selection manifest ready | videos=%d | splits=%s",
            len(selections),
            split_counts,
        )
        return selections

    def _selection_from_row(self, row: dict[str, str], row_number: int) -> DatasetSelection:
        values = {key: (row.get(key) or "").strip() for key in self._REQUIRED_FIELDS}
        if any(not value for value in values.values()):
            raise DatasetIntegrityError(f"manifest row {row_number} contains an empty value")
        if values["split"] not in self._ALLOWED_SPLITS:
            raise DatasetIntegrityError(f"manifest row {row_number} has an invalid split")
        if values["category"] not in self._ALLOWED_CATEGORIES:
            raise DatasetIntegrityError(f"manifest row {row_number} has an invalid category")
        for field in ("subject", "category", "video"):
            if Path(values[field]).name != values[field] or values[field] in {".", ".."}:
                raise DatasetIntegrityError(
                    f"manifest row {row_number} has an unsafe {field} value"
                )
        if Path(values["video"]).suffix.lower() != ".mp4":
            raise DatasetIntegrityError(f"manifest row {row_number} must select an MP4 video")

        coverage = tuple(label.strip() for label in values["coverage"].split("|") if label.strip())
        if not coverage:
            raise DatasetIntegrityError(f"manifest row {row_number} has no coverage labels")
        return DatasetSelection(
            split=values["split"],
            subject=values["subject"],
            category=values["category"],
            video=values["video"],
            coverage=coverage,
            event_focus=values["event_focus"],
        )

    @staticmethod
    def _validate_unique_sources(selections: tuple[DatasetSelection, ...]) -> None:
        identities = [selection.source_relative_path for selection in selections]
        if len(set(identities)) != len(identities):
            raise DatasetIntegrityError("a source video appears more than once in the manifest")

    @staticmethod
    def _validate_subject_isolation(selections: tuple[DatasetSelection, ...]) -> None:
        subject_splits: dict[str, set[str]] = {}
        for selection in selections:
            subject_splits.setdefault(selection.subject, set()).add(selection.split)
        leaked = sorted(subject for subject, splits in subject_splits.items() if len(splits) > 1)
        if leaked:
            raise DatasetIntegrityError(
                f"subjects cannot cross development/evaluation splits: {', '.join(leaked)}"
            )
