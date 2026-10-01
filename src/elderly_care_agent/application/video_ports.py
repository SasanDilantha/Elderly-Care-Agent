"""Dependency-inversion ports for video input adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

from elderly_care_agent.domain.video import TimestampedFrame, VideoMetadata


class VideoSource[FramePayload](ABC):
    """Provides metadata and deterministic indexed frame decoding."""

    @property
    @abstractmethod
    def metadata(self) -> VideoMetadata:
        """Return validated source metadata."""

    @abstractmethod
    def read_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[TimestampedFrame[FramePayload]]:
        """Decode the requested strictly increasing frame indices."""


class VideoSourceFactory[FramePayload](ABC):
    """Creates a source without coupling use cases to OpenCV."""

    @abstractmethod
    def create(self, path: Path) -> VideoSource[FramePayload]:
        """Create a validated source for one video path."""
