"""Immutable domain models and their validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.exceptions import DomainValidationError

_TIME_TOLERANCE_SEC = 1e-6


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A non-empty half-open time interval measured in seconds."""

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if not isfinite(self.start_sec) or not isfinite(self.end_sec):
            raise DomainValidationError("time values must be finite")
        if self.start_sec < 0:
            raise DomainValidationError("start_sec must not be negative")
        if self.end_sec <= self.start_sec:
            raise DomainValidationError("end_sec must be greater than start_sec")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_sec < other.end_sec and other.start_sec < self.end_sec


@dataclass(frozen=True, slots=True)
class StateSegment:
    """A stable activity and bed-occupancy state over a time range."""

    time_range: TimeRange
    activity: ActivityState
    bed_occupancy: BedOccupancy
    confidence: float
    source: ObservationSource

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise DomainValidationError("confidence must be between 0 and 1")
        if (
            self.activity in {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}
            and self.bed_occupancy is BedOccupancy.OUT_OF_BED
        ):
            raise DomainValidationError("an in-bed activity cannot be out of bed")
        if (
            self.activity
            in {
                ActivityState.SITTING_OUTSIDE_BED,
                ActivityState.STANDING,
                ActivityState.WALKING,
            }
            and self.bed_occupancy is BedOccupancy.IN_BED
        ):
            raise DomainValidationError("an outside-bed activity cannot be in bed")


@dataclass(frozen=True, slots=True)
class BedEvent:
    """A confirmed transition across the bed boundary."""

    event_type: EventType
    start_sec: float
    confirmed_sec: float
    previous_activity: ActivityState
    current_activity: ActivityState
    confidence: float
    decision: Decision

    def __post_init__(self) -> None:
        if not isfinite(self.start_sec) or self.start_sec < 0:
            raise DomainValidationError("event start_sec must be finite and non-negative")
        if not isfinite(self.confirmed_sec) or self.confirmed_sec < self.start_sec:
            raise DomainValidationError("confirmed_sec must be at or after start_sec")
        if not 0 <= self.confidence <= 1:
            raise DomainValidationError("event confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class GroundTruthAnnotation:
    """Complete, ordered ground truth for one video."""

    video_id: str
    duration_sec: float
    segments: tuple[StateSegment, ...]
    events: tuple[BedEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise DomainValidationError("video_id must not be empty")
        if not isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise DomainValidationError("duration_sec must be finite and greater than zero")
        if not self.segments:
            raise DomainValidationError("at least one state segment is required")

        first = self.segments[0].time_range
        if not isclose(first.start_sec, 0.0, abs_tol=_TIME_TOLERANCE_SEC):
            raise DomainValidationError("timeline must start at 0 seconds")

        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if not isclose(
                previous.time_range.end_sec,
                current.time_range.start_sec,
                abs_tol=_TIME_TOLERANCE_SEC,
            ):
                raise DomainValidationError("timeline segments must be ordered and contiguous")

        if not isclose(
            self.segments[-1].time_range.end_sec,
            self.duration_sec,
            abs_tol=_TIME_TOLERANCE_SEC,
        ):
            raise DomainValidationError("timeline must end at duration_sec")

        for event in self.events:
            if event.confirmed_sec > self.duration_sec + _TIME_TOLERANCE_SEC:
                raise DomainValidationError("event must occur within the video duration")

    def activity_durations(self) -> dict[ActivityState, float]:
        """Aggregate seconds spent in each activity present in the timeline."""

        durations: dict[ActivityState, float] = {}
        for segment in self.segments:
            durations[segment.activity] = (
                durations.get(segment.activity, 0.0) + segment.time_range.duration_sec
            )
        return durations

    def occupancy_durations(self) -> dict[BedOccupancy, float]:
        """Aggregate seconds spent in each occupancy state."""

        durations: dict[BedOccupancy, float] = {}
        for segment in self.segments:
            durations[segment.bed_occupancy] = (
                durations.get(segment.bed_occupancy, 0.0) + segment.time_range.duration_sec
            )
        return durations
