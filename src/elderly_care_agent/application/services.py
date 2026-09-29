"""Small application services that orchestrate domain behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elderly_care_agent.application.ports import AnnotationRepository


@dataclass(frozen=True, slots=True)
class ValidationReport:
    video_id: str
    duration_sec: float
    segment_count: int
    event_count: int
    activity_durations_sec: dict[str, float]
    occupancy_durations_sec: dict[str, float]


class AnnotationValidationService:
    """Validates an annotation and returns a compact, UI-neutral report."""

    def __init__(self, repository: AnnotationRepository) -> None:
        self._repository = repository

    def validate(self, path: Path) -> ValidationReport:
        annotation = self._repository.load(path)
        return ValidationReport(
            video_id=annotation.video_id,
            duration_sec=annotation.duration_sec,
            segment_count=len(annotation.segments),
            event_count=len(annotation.events),
            activity_durations_sec={
                state.value: duration
                for state, duration in annotation.activity_durations().items()
            },
            occupancy_durations_sec={
                state.value: duration
                for state, duration in annotation.occupancy_durations().items()
            },
        )

