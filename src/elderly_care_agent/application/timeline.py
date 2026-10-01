"""Turn frame-level rule evidence into a complete, stable timeline."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from elderly_care_agent.application.rules import RuleBasedStateClassifier, RuleObservation
from elderly_care_agent.application.vision_features import VisionAnalysisService
from elderly_care_agent.config import TemporalSettings
from elderly_care_agent.domain.enums import ActivityState, BedOccupancy, ObservationSource
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import StateSegment, StateTimeline, TimeRange
from elderly_care_agent.domain.vision import BedRegion

logger = logging.getLogger(__name__)


class TemporalStateSmoother:
    """Accepts sustained rules; brief or missing evidence becomes UNKNOWN."""

    def __init__(self, confirmation_sec: float) -> None:
        if not isfinite(confirmation_sec) or confirmation_sec <= 0:
            raise ValueError("confirmation_sec must be finite and positive")
        self._confirmation_sec = confirmation_sec

    def build(
        self,
        video_id: str,
        duration_sec: float,
        observations: tuple[RuleObservation, ...],
    ) -> StateTimeline:
        self._validate_observations(duration_sec, observations)
        if not observations:
            return StateTimeline(
                video_id,
                duration_sec,
                (self._unknown_segment(0.0, duration_sec),),
            )

        segments: list[StateSegment] = []
        index = 0
        while index < len(observations):
            first = observations[index]
            stop = index + 1
            while stop < len(observations) and self._same_state(first, observations[stop]):
                stop += 1
            start_sec = first.reference.timestamp_sec
            end_sec = (
                observations[stop].reference.timestamp_sec
                if stop < len(observations)
                else duration_sec
            )
            if (
                first.activity is not ActivityState.UNKNOWN
                and stop - index >= 2
                and end_sec - start_sec >= self._confirmation_sec
            ):
                confidence = sum(item.confidence for item in observations[index:stop]) / (
                    stop - index
                )
                segment = StateSegment(
                    TimeRange(start_sec, end_sec),
                    first.activity,
                    first.bed_occupancy,
                    confidence,
                    ObservationSource.RULES,
                )
            else:
                segment = self._unknown_segment(start_sec, end_sec)
            self._append_or_merge(segments, segment)
            index = stop

        return StateTimeline(video_id, duration_sec, tuple(segments))

    @staticmethod
    def _validate_observations(
        duration_sec: float, observations: tuple[RuleObservation, ...]
    ) -> None:
        if not isfinite(duration_sec) or duration_sec <= 0:
            raise DomainValidationError("duration_sec must be finite and greater than zero")
        if not observations:
            return
        if not isclose(observations[0].reference.timestamp_sec, 0.0, abs_tol=1e-6):
            raise DomainValidationError("first observation must start at 0 seconds")
        previous_time = -1.0
        for observation in observations:
            time = observation.reference.timestamp_sec
            if time <= previous_time or time >= duration_sec:
                raise DomainValidationError("observations must be ordered within video duration")
            previous_time = time

    @staticmethod
    def _same_state(first: RuleObservation, second: RuleObservation) -> bool:
        return first.activity is second.activity and first.bed_occupancy is second.bed_occupancy

    @staticmethod
    def _unknown_segment(start_sec: float, end_sec: float) -> StateSegment:
        return StateSegment(
            TimeRange(start_sec, end_sec),
            ActivityState.UNKNOWN,
            BedOccupancy.UNKNOWN,
            0.0,
            ObservationSource.RULES,
        )

    @staticmethod
    def _append_or_merge(segments: list[StateSegment], segment: StateSegment) -> None:
        if (
            segments
            and segments[-1].activity is segment.activity
            and (segments[-1].bed_occupancy is segment.bed_occupancy)
        ):
            previous = segments.pop()
            segments.append(
                StateSegment(
                    TimeRange(previous.time_range.start_sec, segment.time_range.end_sec),
                    segment.activity,
                    segment.bed_occupancy,
                    0.0 if segment.activity is ActivityState.UNKNOWN else segment.confidence,
                    ObservationSource.RULES,
                )
            )
        else:
            segments.append(segment)


@dataclass(frozen=True, slots=True)
class TimelineAnalysisReport:
    video_id: str
    source_path: str
    duration_sec: float
    sample_fps: float
    bed_region: BedRegion | None
    sampled_frame_count: int
    observations: tuple[RuleObservation, ...]
    segments: tuple[StateSegment, ...]
    activity_durations_sec: dict[str, float]
    occupancy_durations_sec: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TimelineAnalysisService[FramePayload]:
    """Runs vision once, classifies frames, then aggregates stable segments."""

    def __init__(
        self,
        vision_service: VisionAnalysisService[FramePayload],
        temporal_settings: TemporalSettings,
        classifier: RuleBasedStateClassifier | None = None,
    ) -> None:
        self._vision_service = vision_service
        self._classifier = classifier or RuleBasedStateClassifier()
        self._smoother = TemporalStateSmoother(temporal_settings.state_confirmation_sec)

    def analyze(
        self,
        path: Path,
        sample_fps: float | None = None,
        bed_region: BedRegion | None = None,
    ) -> TimelineAnalysisReport:
        vision = self._vision_service.analyze(path, sample_fps, bed_region)
        observations = self._classifier.classify_all(vision.frames, vision.width, vision.height)
        timeline = self._smoother.build(vision.video_id, vision.duration_sec, observations)
        report = TimelineAnalysisReport(
            video_id=vision.video_id,
            source_path=vision.source_path,
            duration_sec=vision.duration_sec,
            sample_fps=vision.sample_fps,
            bed_region=vision.bed_region,
            sampled_frame_count=len(observations),
            observations=observations,
            segments=timeline.segments,
            activity_durations_sec={
                state.value: seconds for state, seconds in timeline.activity_durations().items()
            },
            occupancy_durations_sec={
                state.value: seconds for state, seconds in timeline.occupancy_durations().items()
            },
        )
        logger.info(
            "Timeline analysis completed | video=%s | frames=%d | segments=%d | unknown_sec=%.3f",
            report.video_id,
            report.sampled_frame_count,
            len(report.segments),
            report.activity_durations_sec.get(ActivityState.UNKNOWN.value, 0.0),
        )
        return report
