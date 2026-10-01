"""Framework-independent frame observations and bed geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.video import FrameReference


class PersonStatus(StrEnum):
    DETECTED = "detected"
    UNKNOWN = "unknown"


class BedRelation(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) and 0 <= value <= 1 for value in (self.x, self.y)):
            raise DomainValidationError("point coordinates must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class BedRegion:
    """Camera-specific rectangle in normalized image coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(isfinite(value) and 0 <= value <= 1 for value in values):
            raise DomainValidationError("bed region coordinates must be within [0, 1]")
        if self.left >= self.right or self.top >= self.bottom:
            raise DomainValidationError("bed region must have positive width and height")

    @classmethod
    def parse(cls, value: str) -> BedRegion:
        try:
            coordinates = tuple(float(item.strip()) for item in value.split(","))
        except ValueError as error:
            raise DomainValidationError("bed region must be left,top,right,bottom") from error
        if len(coordinates) != 4:
            raise DomainValidationError("bed region must be left,top,right,bottom")
        return cls(*coordinates)

    def contains(self, point: NormalizedPoint) -> bool:
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom


@dataclass(frozen=True, slots=True)
class PoseLandmark:
    """One MediaPipe landmark with normalized image coordinates."""

    index: int
    x: float
    y: float
    z: float
    visibility: float
    presence: float

    def __post_init__(self) -> None:
        if not 0 <= self.index < 33:
            raise DomainValidationError("pose landmark index must be within [0, 32]")
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise DomainValidationError("pose landmark coordinates must be finite")
        if not all(
            isfinite(value) and 0 <= value <= 1 for value in (self.visibility, self.presence)
        ):
            raise DomainValidationError("pose landmark confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class VisionFrame:
    """Geometry only; later stages decide activity and bed occupancy."""

    reference: FrameReference
    person_status: PersonStatus
    bed_relation: BedRelation
    body_anchor: NormalizedPoint | None
    torso_angle_deg: float | None
    landmarks: tuple[PoseLandmark, ...]
    unknown_reason: str | None = None
