"""Immutable entities used by the GMDCSA-24 preparation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Pinned identity and licensing information for an external dataset."""

    owner: str
    canonical_repository: str
    mirror_repository: str
    mirror_commit: str
    archive_url: str
    archive_checksum: str
    dataset_doi: str
    article_doi: str
    dataset_license: str
    repository_license: str


@dataclass(frozen=True, slots=True)
class DatasetIngestionConfig:
    """Resolved filesystem and integrity settings for one pipeline run."""

    dataset_name: str
    dataset_version: str
    source: DatasetSource
    archive_path: Path
    source_dir: Path
    selection_manifest: Path
    output_dir: Path
    report_path: Path
    inventory_path: Path
    expected_video_count: int
    expected_metadata_csv_count: int
    expected_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """One explicitly selected source video and its intended split."""

    split: str
    subject: str
    category: str
    video: str
    coverage: tuple[str, ...]
    event_focus: str

    @property
    def source_relative_path(self) -> Path:
        return Path(self.subject, self.category, self.video)

    @property
    def destination_relative_path(self) -> Path:
        return Path(self.split, self.subject, self.video)


@dataclass(frozen=True, slots=True)
class VideoInventoryItem:
    """Verified metadata and identity of one staged video."""

    split: str
    subject: str
    category: str
    video: str
    relative_path: str
    sha256: str
    size_bytes: int
    frame_count: int
    fps: float
    duration_sec: float


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Minimal decoder-backed metadata proving that a video can be opened."""

    frame_count: int
    fps: float
    duration_sec: float


@dataclass(frozen=True, slots=True)
class DatasetPreparationReport:
    """Machine-readable outcome of a reproducible dataset preparation run."""

    status: str
    dataset_name: str
    dataset_version: str
    source_owner: str
    source_doi: str
    source_repository: str
    mirror_repository: str
    mirror_commit: str
    dataset_license: str
    source_mode: str
    manifest_sha256: str
    selected_video_count: int
    copied_video_count: int
    reused_video_count: int
    total_size_bytes: int
    split_counts: dict[str, int]
    report_path: str
    inventory_path: str
    completed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)
