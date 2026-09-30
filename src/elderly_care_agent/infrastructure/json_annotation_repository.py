"""JSON adapter for the annotation repository port."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elderly_care_agent.application.ports import AnnotationRepository
from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.exceptions import AnnotationFormatError
from elderly_care_agent.domain.models import (
    BedEvent,
    GroundTruthAnnotation,
    StateSegment,
    TimeRange,
)


class JsonAnnotationRepository(AnnotationRepository):
    """Converts the documented JSON schema into validated domain objects."""

    def load(self, path: Path) -> GroundTruthAnnotation:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._to_annotation(payload)
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationFormatError(f"cannot read annotation '{path}': {error}") from error
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationFormatError(f"invalid annotation '{path}': {error}") from error

    def _to_annotation(self, payload: dict[str, Any]) -> GroundTruthAnnotation:
        segments = tuple(self._to_segment(item) for item in payload["segments"])
        events = tuple(self._to_event(item) for item in payload.get("events", []))
        return GroundTruthAnnotation(
            video_id=str(payload["video_id"]),
            duration_sec=float(payload["duration_sec"]),
            segments=segments,
            events=events,
        )

    @staticmethod
    def _to_segment(payload: dict[str, Any]) -> StateSegment:
        return StateSegment(
            time_range=TimeRange(
                start_sec=float(payload["start_sec"]),
                end_sec=float(payload["end_sec"]),
            ),
            activity=ActivityState(payload["activity"]),
            bed_occupancy=BedOccupancy(payload["bed_occupancy"]),
            confidence=float(payload.get("confidence", 1.0)),
            source=ObservationSource(payload.get("source", "ground_truth")),
        )

    @staticmethod
    def _to_event(payload: dict[str, Any]) -> BedEvent:
        return BedEvent(
            event_type=EventType(payload["event_type"]),
            start_sec=float(payload["start_sec"]),
            confirmed_sec=float(payload["confirmed_sec"]),
            previous_activity=ActivityState(payload["previous_activity"]),
            current_activity=ActivityState(payload["current_activity"]),
            confidence=float(payload.get("confidence", 1.0)),
            decision=Decision(payload["decision"]),
        )
