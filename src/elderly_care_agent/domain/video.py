"""Framework-neutral video metadata and timestamped frame entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from elderly_care_agent.domain.exceptions import DomainValidationError


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Stable information required to sample and report on one video."""

    video_id: str
    source_path: str
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise DomainValidationError("video_id must not be empty")
        if not self.source_path.strip():
            raise DomainValidationError("source_path must not be empty")
        if not isfinite(self.fps) or self.fps <= 0:
            raise DomainValidationError("fps must be finite and greater than zero")
        if self.frame_count <= 0:
            raise DomainValidationError("frame_count must be greater than zero")
        if self.width <= 0 or self.height <= 0:
            raise DomainValidationError("video dimensions must be greater than zero")

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class TimestampedFrame[FramePayload]:
    """A decoded payload tied to its original frame index and media time."""

    frame_index: int
    timestamp_sec: float
    payload: FramePayload = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DomainValidationError("frame_index must not be negative")
        if not isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise DomainValidationError("timestamp_sec must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Serialization-safe identity of a sampled frame."""

    frame_index: int
    timestamp_sec: float

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise DomainValidationError("frame_index must not be negative")
        if not isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise DomainValidationError("timestamp_sec must be finite and non-negative")
