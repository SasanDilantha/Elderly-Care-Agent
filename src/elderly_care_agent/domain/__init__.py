"""Domain types for activities, occupancy, events, and timelines."""

from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.models import (
    BedEvent,
    GroundTruthAnnotation,
    StateSegment,
    TimeRange,
)
from elderly_care_agent.domain.video import FrameReference, TimestampedFrame, VideoMetadata

__all__ = [
    "ActivityState",
    "BedEvent",
    "BedOccupancy",
    "Decision",
    "EventType",
    "FrameReference",
    "GroundTruthAnnotation",
    "ObservationSource",
    "StateSegment",
    "TimeRange",
    "TimestampedFrame",
    "VideoMetadata",
]
