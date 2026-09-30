"""Dependency-inversion ports used by dataset preparation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from elderly_care_agent.domain.dataset import DatasetSelection, VideoProbeResult


class DatasetDownloader(ABC):
    """Downloads an immutable dataset archive."""

    @abstractmethod
    def download(self, source_url: str, destination: Path) -> None:
        """Download to destination atomically."""


class DatasetExtractor(ABC):
    """Extracts a dataset archive without trusting member paths."""

    @abstractmethod
    def extract(self, archive_path: Path, destination: Path) -> None:
        """Safely extract archive contents into an empty destination."""


class DatasetSelectionRepository(ABC):
    """Loads the version-controlled subset declaration."""

    @abstractmethod
    def load(self, manifest_path: Path) -> tuple[DatasetSelection, ...]:
        """Load and validate all selected videos."""


class VideoProbe(ABC):
    """Proves that a selected video is decoder-readable."""

    @abstractmethod
    def inspect(self, video_path: Path) -> VideoProbeResult:
        """Read a frame and return stable metadata."""
