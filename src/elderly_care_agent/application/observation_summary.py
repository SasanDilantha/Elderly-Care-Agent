"""Video-level durations and explainable contextual decisions."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Any

from elderly_care_agent.application.bed_events import (
    BedEventAnalysisReport,
    BedEventAnalysisService,
)
from elderly_care_agent.config import DecisionSettings
from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.models import BedEvent, StateSegment
from elderly_care_agent.domain.vision import BedRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OccupancyRun:
    occupancy: BedOccupancy
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class OccupancyRunGrouper:
    """Join contiguous occupancy intervals without crossing unknown evidence."""

    def group(
        self,
        segments: tuple[StateSegment, ...],
        *,
        rules_only: bool = False,
    ) -> tuple[OccupancyRun, ...]:
        runs: list[OccupancyRun] = []
        for segment in segments:
            if rules_only and segment.source is not ObservationSource.RULES:
                continue
            current = segment.time_range
            if (
                runs
                and runs[-1].occupancy is segment.bed_occupancy
                and isclose(runs[-1].end_sec, current.start_sec, abs_tol=1e-6)
            ):
                previous = runs.pop()
                runs.append(
                    OccupancyRun(segment.bed_occupancy, previous.start_sec, current.end_sec)
                )
            else:
                runs.append(OccupancyRun(segment.bed_occupancy, current.start_sec, current.end_sec))
        return tuple(runs)


@dataclass(frozen=True, slots=True)
class DecisionAssessment:
    decision: Decision
    reason: str
    trigger_sec: float | None


class ContextualDecisionPolicy:
    """Never alert without a confirmed exit and continuous rule evidence."""

    def __init__(
        self,
        settings: DecisionSettings,
        grouper: OccupancyRunGrouper | None = None,
    ) -> None:
        self._settings = settings
        self._grouper = grouper or OccupancyRunGrouper()

    def assess(self, report: BedEventAnalysisReport) -> DecisionAssessment:
        trusted_runs = self._grouper.group(report.fused_segments, rules_only=True)
        exits = (event for event in report.events if event.event_type is EventType.BED_EXIT)
        for event in exits:
            for run in trusted_runs:
                if (
                    run.occupancy is BedOccupancy.OUT_OF_BED
                    and isclose(run.start_sec, event.start_sec, abs_tol=1e-6)
                    and run.duration_sec >= self._settings.prolonged_absence_sec
                ):
                    return DecisionAssessment(
                        Decision.ALERT,
                        "confirmed_prolonged_absence_from_bed",
                        max(
                            event.confirmed_sec,
                            event.start_sec + self._settings.prolonged_absence_sec,
                        ),
                    )

        monitor_triggers = [
            (event.confirmed_sec, "confirmed_bed_exit")
            for event in report.events
            if event.event_type is EventType.BED_EXIT
        ]
        for run in self._grouper.group(report.fused_segments):
            if (
                run.occupancy is BedOccupancy.UNKNOWN
                and run.duration_sec >= self._settings.uncertain_monitor_sec
            ):
                monitor_triggers.append(
                    (run.start_sec + self._settings.uncertain_monitor_sec, "prolonged_unknown")
                )
            if (
                run.occupancy is BedOccupancy.OUT_OF_BED
                and run.duration_sec >= self._settings.prolonged_absence_sec
                and not any(
                    event.event_type is EventType.BED_EXIT
                    and isclose(event.start_sec, run.start_sec, abs_tol=1e-6)
                    for event in report.events
                )
            ):
                monitor_triggers.append(
                    (
                        run.start_sec + self._settings.prolonged_absence_sec,
                        "prolonged_out_of_bed_without_confirmed_exit",
                    )
                )
        if monitor_triggers:
            trigger_sec, reason = min(monitor_triggers, key=lambda item: item[0])
            return DecisionAssessment(Decision.MONITOR, reason, trigger_sec)
        return DecisionAssessment(Decision.NORMAL, "no_policy_trigger", None)


@dataclass(frozen=True, slots=True)
class ObservationSummary:
    video_id: str
    source_path: str
    observation_duration_sec: float
    vlm_enabled: bool
    reviewed_segment_count: int
    promoted_vlm_segment_count: int
    activity_duration_sec: dict[str, float]
    occupancy_duration_sec: dict[str, float]
    total_in_bed_sec: float
    total_out_of_bed_sec: float
    total_unknown_bed_sec: float
    longest_out_of_bed_period_sec: float
    bed_exit_count: int
    bed_return_count: int
    final_activity: ActivityState
    final_bed_occupancy: BedOccupancy
    decision: DecisionAssessment
    timeline: tuple[StateSegment, ...]
    events: tuple[BedEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationSummaryService[FramePayload]:
    """Builds a complete summary from one bed-event analysis run."""

    def __init__(
        self,
        event_service: BedEventAnalysisService[FramePayload],
        decision_settings: DecisionSettings,
        grouper: OccupancyRunGrouper | None = None,
        policy: ContextualDecisionPolicy | None = None,
    ) -> None:
        self._event_service = event_service
        self._grouper = grouper or OccupancyRunGrouper()
        self._policy = policy or ContextualDecisionPolicy(decision_settings, self._grouper)

    def summarize(
        self,
        path: Path,
        sample_fps: float | None = None,
        bed_region: BedRegion | None = None,
        *,
        with_vlm: bool = False,
        max_segments: int | None = None,
    ) -> ObservationSummary:
        report = self._event_service.analyze(
            path,
            sample_fps,
            bed_region,
            with_vlm=with_vlm,
            max_segments=max_segments,
        )
        out_runs = (
            run
            for run in self._grouper.group(report.fused_segments)
            if run.occupancy is BedOccupancy.OUT_OF_BED
        )
        assessment = self._policy.assess(report)
        summary = ObservationSummary(
            video_id=report.video_id,
            source_path=report.source_path,
            observation_duration_sec=report.duration_sec,
            vlm_enabled=report.vlm_enabled,
            reviewed_segment_count=len(report.vlm_reviews),
            promoted_vlm_segment_count=report.promoted_vlm_segment_count,
            activity_duration_sec=report.activity_durations_sec,
            occupancy_duration_sec=report.occupancy_durations_sec,
            total_in_bed_sec=report.occupancy_durations_sec.get(BedOccupancy.IN_BED.value, 0.0),
            total_out_of_bed_sec=report.occupancy_durations_sec.get(
                BedOccupancy.OUT_OF_BED.value, 0.0
            ),
            total_unknown_bed_sec=report.occupancy_durations_sec.get(
                BedOccupancy.UNKNOWN.value, 0.0
            ),
            longest_out_of_bed_period_sec=max((run.duration_sec for run in out_runs), default=0.0),
            bed_exit_count=sum(event.event_type is EventType.BED_EXIT for event in report.events),
            bed_return_count=sum(
                event.event_type is EventType.BED_RETURN for event in report.events
            ),
            final_activity=report.fused_segments[-1].activity,
            final_bed_occupancy=report.fused_segments[-1].bed_occupancy,
            decision=assessment,
            timeline=report.fused_segments,
            events=report.events,
        )
        logger.info(
            "Observation summarized | video=%s | decision=%s | reason=%s | "
            "exits=%d | returns=%d | out_sec=%.3f",
            summary.video_id,
            summary.decision.decision,
            summary.decision.reason,
            summary.bed_exit_count,
            summary.bed_return_count,
            summary.total_out_of_bed_sec,
        )
        return summary
