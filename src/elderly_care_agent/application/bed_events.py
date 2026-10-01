"""Conservative timeline fusion and confirmed bed-boundary events."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from elderly_care_agent.application.timeline import TimelineAnalysisService
from elderly_care_agent.application.vlm_review import VlmReviewService
from elderly_care_agent.config import TemporalSettings
from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import BedEvent, StateSegment, StateTimeline
from elderly_care_agent.domain.vision import BedRegion
from elderly_care_agent.domain.vlm import ReviewStatus, VlmReview

logger = logging.getLogger(__name__)


class ConservativeTimelineFusion:
    """Fill only rule-unknown gaps enclosed by matching known occupancy."""

    def __init__(self, minimum_confidence: float = 0.6) -> None:
        if not isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be within [0, 1]")
        self._minimum_confidence = minimum_confidence

    def fuse(
        self,
        timeline: StateTimeline,
        reviews: tuple[VlmReview, ...],
    ) -> StateTimeline:
        by_range = {review.time_range: review for review in reviews}
        if len(by_range) != len(reviews):
            raise DomainValidationError("VLM reviews must have unique intervals")
        rule_ranges = {segment.time_range for segment in timeline.segments}
        if not set(by_range).issubset(rule_ranges):
            raise DomainValidationError("VLM review interval is absent from the rule timeline")

        fused: list[StateSegment] = []
        segments = timeline.segments
        for index, segment in enumerate(segments):
            review = by_range.get(segment.time_range)
            if review is None:
                fused.append(segment)
                continue
            if segment.activity is not ActivityState.UNKNOWN:
                raise DomainValidationError("VLM review may only target a rule-unknown interval")
            if review.source is not ObservationSource.VLM:
                raise DomainValidationError("VLM review must have VLM provenance")
            if (
                review.status is ReviewStatus.PROPOSED
                and review.self_reported_confidence >= self._minimum_confidence
                and 0 < index < len(segments) - 1
                and segments[index - 1].bed_occupancy is review.bed_occupancy
                and segments[index + 1].bed_occupancy is review.bed_occupancy
                and review.bed_occupancy is not BedOccupancy.UNKNOWN
            ):
                fused.append(
                    StateSegment(
                        segment.time_range,
                        review.activity,
                        review.bed_occupancy,
                        review.self_reported_confidence,
                        ObservationSource.FUSED,
                    )
                )
            else:
                fused.append(segment)
        return StateTimeline(timeline.video_id, timeline.duration_sec, tuple(fused))


class BedTransitionDetector:
    """Confirm direct rule-supported occupancy changes after a dwell period."""

    def __init__(self, settings: TemporalSettings) -> None:
        self._state_confirmation_sec = settings.state_confirmation_sec
        self._event_confirmation_sec = settings.bed_event_confirmation_sec

    def detect(self, timeline: StateTimeline) -> tuple[BedEvent, ...]:
        events: list[BedEvent] = []
        for previous, current in zip(timeline.segments, timeline.segments[1:], strict=False):
            if (
                previous.source is not ObservationSource.RULES
                or current.source is not ObservationSource.RULES
                or previous.time_range.duration_sec < self._state_confirmation_sec
                or current.time_range.duration_sec < self._event_confirmation_sec
            ):
                continue
            if (
                previous.bed_occupancy is BedOccupancy.IN_BED
                and current.bed_occupancy is BedOccupancy.OUT_OF_BED
            ):
                event_type, decision = EventType.BED_EXIT, Decision.MONITOR
            elif (
                previous.bed_occupancy is BedOccupancy.OUT_OF_BED
                and current.bed_occupancy is BedOccupancy.IN_BED
            ):
                event_type, decision = EventType.BED_RETURN, Decision.NORMAL
            else:
                continue
            start = current.time_range.start_sec
            confirmed = start + self._event_confirmation_sec
            if confirmed > timeline.duration_sec and not isclose(confirmed, timeline.duration_sec):
                continue
            events.append(
                BedEvent(
                    event_type=event_type,
                    start_sec=start,
                    confirmed_sec=confirmed,
                    previous_activity=previous.activity,
                    current_activity=current.activity,
                    confidence=min(previous.confidence, current.confidence),
                    decision=decision,
                )
            )
        return tuple(events)


@dataclass(frozen=True, slots=True)
class BedEventAnalysisReport:
    video_id: str
    source_path: str
    duration_sec: float
    vlm_enabled: bool
    rule_segments: tuple[StateSegment, ...]
    vlm_reviews: tuple[VlmReview, ...]
    fused_segments: tuple[StateSegment, ...]
    events: tuple[BedEvent, ...]
    activity_durations_sec: dict[str, float]
    occupancy_durations_sec: dict[str, float]

    @property
    def promoted_vlm_segment_count(self) -> int:
        return sum(segment.source is ObservationSource.FUSED for segment in self.fused_segments)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "promoted_vlm_segment_count": self.promoted_vlm_segment_count,
            "event_count": self.event_count,
        }


class BedEventAnalysisService[FramePayload]:
    """Runs rules once, optionally reviews gaps, then emits safe events."""

    def __init__(
        self,
        timeline_service: TimelineAnalysisService[FramePayload],
        settings: TemporalSettings,
        review_service: VlmReviewService[FramePayload] | None = None,
        fusion: ConservativeTimelineFusion | None = None,
        detector: BedTransitionDetector | None = None,
    ) -> None:
        self._timeline_service = timeline_service
        self._review_service = review_service
        self._fusion = fusion or ConservativeTimelineFusion()
        self._detector = detector or BedTransitionDetector(settings)

    def analyze(
        self,
        path: Path,
        sample_fps: float | None = None,
        bed_region: BedRegion | None = None,
        *,
        with_vlm: bool = False,
        max_segments: int | None = None,
    ) -> BedEventAnalysisReport:
        if max_segments is not None and not with_vlm:
            raise DomainValidationError("max_segments requires with_vlm")
        if with_vlm:
            if self._review_service is None:
                raise DomainValidationError("VLM review service is unavailable")
            review_report = self._review_service.review(path, sample_fps, bed_region, max_segments)
            timeline = StateTimeline(
                review_report.video_id,
                review_report.duration_sec,
                review_report.rule_segments,
            )
            source_path = review_report.source_path
            reviews = review_report.reviews
        else:
            rule_report = self._timeline_service.analyze(path, sample_fps, bed_region)
            timeline = StateTimeline(
                rule_report.video_id,
                rule_report.duration_sec,
                rule_report.segments,
            )
            source_path = rule_report.source_path
            reviews = ()

        fused = self._fusion.fuse(timeline, reviews)
        events = self._detector.detect(fused)
        logger.info(
            "Bed-event analysis completed | video=%s | vlm=%s | "
            "reviews=%d | promoted=%d | events=%d",
            timeline.video_id,
            with_vlm,
            len(reviews),
            sum(segment.source is ObservationSource.FUSED for segment in fused.segments),
            len(events),
        )
        return BedEventAnalysisReport(
            video_id=timeline.video_id,
            source_path=source_path,
            duration_sec=timeline.duration_sec,
            vlm_enabled=with_vlm,
            rule_segments=timeline.segments,
            vlm_reviews=reviews,
            fused_segments=fused.segments,
            events=events,
            activity_durations_sec={
                state.value: duration for state, duration in fused.activity_durations().items()
            },
            occupancy_durations_sec={
                state.value: duration for state, duration in fused.occupancy_durations().items()
            },
        )
