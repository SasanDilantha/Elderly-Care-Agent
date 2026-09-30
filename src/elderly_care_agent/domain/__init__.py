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

__all__ = [
    "ActivityState",
    "BedEvent",
    "BedOccupancy",
    "Decision",
    "EventType",
    "GroundTruthAnnotation",
    "ObservationSource",
    "StateSegment",
    "TimeRange",
]
