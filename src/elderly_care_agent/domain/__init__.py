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
from elderly_care_agent.domain.vision import (
    BedRegion,
    BedRelation,
    NormalizedPoint,
    PersonStatus,
    PoseLandmark,
    VisionFrame,
)

__all__ = [
    "ActivityState",
    "BedEvent",
    "BedRegion",
    "BedRelation",
    "BedOccupancy",
    "Decision",
    "EventType",
    "FrameReference",
    "GroundTruthAnnotation",
    "ObservationSource",
    "NormalizedPoint",
    "PersonStatus",
    "PoseLandmark",
    "StateSegment",
    "TimeRange",
    "TimestampedFrame",
    "VideoMetadata",
    "VisionFrame",
]
