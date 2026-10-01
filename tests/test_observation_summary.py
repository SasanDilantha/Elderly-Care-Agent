import unittest
from pathlib import Path
from unittest.mock import Mock

from elderly_care_agent.application.bed_events import BedEventAnalysisReport
from elderly_care_agent.application.observation_summary import ObservationSummaryService
from elderly_care_agent.config import DecisionSettings
from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.models import BedEvent, StateSegment, StateTimeline, TimeRange


def state(
    start: float,
    end: float,
    activity: ActivityState,
    occupancy: BedOccupancy,
    source: ObservationSource = ObservationSource.RULES,
) -> StateSegment:
    confidence = 0.0 if activity is ActivityState.UNKNOWN else 0.7
    return StateSegment(TimeRange(start, end), activity, occupancy, confidence, source)


def exit_event(start: float, confirmed: float) -> BedEvent:
    return BedEvent(
        EventType.BED_EXIT,
        start,
        confirmed,
        ActivityState.SITTING_ON_BED,
        ActivityState.STANDING,
        0.7,
        Decision.MONITOR,
    )


def return_event(start: float, confirmed: float) -> BedEvent:
    return BedEvent(
        EventType.BED_RETURN,
        start,
        confirmed,
        ActivityState.STANDING,
        ActivityState.SITTING_ON_BED,
        0.7,
        Decision.NORMAL,
    )


def report(*segments: StateSegment, events: tuple[BedEvent, ...] = ()) -> BedEventAnalysisReport:
    timeline = StateTimeline("clip.mp4", segments[-1].time_range.end_sec, segments)
    return BedEventAnalysisReport(
        video_id="clip.mp4",
        source_path="clip.mp4",
        duration_sec=timeline.duration_sec,
        vlm_enabled=False,
        rule_segments=segments,
        vlm_reviews=(),
        fused_segments=segments,
        events=events,
        activity_durations_sec={
            activity.value: duration for activity, duration in timeline.activity_durations().items()
        },
        occupancy_durations_sec={
            occupancy.value: duration
            for occupancy, duration in timeline.occupancy_durations().items()
        },
    )


class ObservationSummaryTests(unittest.TestCase):
    def summarize(self, analysis: BedEventAnalysisReport):
        event_service = Mock()
        event_service.analyze.return_value = analysis
        summary = ObservationSummaryService(event_service, DecisionSettings()).summarize(
            Path("clip.mp4")
        )
        event_service.analyze.assert_called_once_with(
            Path("clip.mp4"), None, None, with_vlm=False, max_segments=None
        )
        return summary

    def test_normal_summary_has_complete_durations_and_final_state(self) -> None:
        summary = self.summarize(
            report(state(0, 20, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED))
        )

        self.assertEqual(Decision.NORMAL, summary.decision.decision)
        self.assertEqual("no_policy_trigger", summary.decision.reason)
        self.assertIsNone(summary.decision.trigger_sec)
        self.assertEqual(20, summary.observation_duration_sec)
        self.assertEqual(20, summary.total_in_bed_sec)
        self.assertEqual(0, summary.total_out_of_bed_sec)
        self.assertEqual(0, summary.bed_exit_count)
        self.assertEqual(ActivityState.LYING_IN_BED, summary.final_activity)
        self.assertEqual(20, sum(summary.activity_duration_sec.values()))

    def test_confirmed_exit_is_monitor_before_absence_threshold(self) -> None:
        summary = self.summarize(
            report(
                state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
                state(3, 20, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
                events=(exit_event(3, 5),),
            )
        )

        self.assertEqual(Decision.MONITOR, summary.decision.decision)
        self.assertEqual("confirmed_bed_exit", summary.decision.reason)
        self.assertEqual(5, summary.decision.trigger_sec)
        self.assertEqual(17, summary.longest_out_of_bed_period_sec)
        self.assertEqual(1, summary.bed_exit_count)

    def test_prolonged_rule_supported_absence_alerts_at_threshold(self) -> None:
        summary = self.summarize(
            report(
                state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
                state(3, 30, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
                state(30, 63, ActivityState.WALKING, BedOccupancy.OUT_OF_BED),
                events=(exit_event(3, 5),),
            )
        )

        self.assertEqual(Decision.ALERT, summary.decision.decision)
        self.assertEqual("confirmed_prolonged_absence_from_bed", summary.decision.reason)
        self.assertEqual(63, summary.decision.trigger_sec)
        self.assertEqual(60, summary.longest_out_of_bed_period_sec)
        self.assertEqual(60, summary.total_out_of_bed_sec)

    def test_out_of_bed_at_video_start_is_monitor_not_invented_alert(self) -> None:
        summary = self.summarize(
            report(state(0, 90, ActivityState.WALKING, BedOccupancy.OUT_OF_BED))
        )

        self.assertEqual(Decision.MONITOR, summary.decision.decision)
        self.assertEqual("prolonged_out_of_bed_without_confirmed_exit", summary.decision.reason)
        self.assertEqual(60, summary.decision.trigger_sec)
        self.assertEqual(90, summary.longest_out_of_bed_period_sec)
        self.assertEqual(0, summary.bed_exit_count)

    def test_two_short_absences_do_not_add_up_to_an_alert(self) -> None:
        summary = self.summarize(
            report(
                state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
                state(3, 38, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
                state(38, 43, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
                state(43, 78, ActivityState.WALKING, BedOccupancy.OUT_OF_BED),
                events=(exit_event(3, 5), return_event(38, 40), exit_event(43, 45)),
            )
        )

        self.assertEqual(Decision.MONITOR, summary.decision.decision)
        self.assertEqual(70, summary.total_out_of_bed_sec)
        self.assertEqual(35, summary.longest_out_of_bed_period_sec)
        self.assertEqual(2, summary.bed_exit_count)
        self.assertEqual(1, summary.bed_return_count)

    def test_unknown_and_vlm_gap_interrupt_alert_evidence(self) -> None:
        for gap in (
            state(30, 31, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN),
            state(
                30,
                31,
                ActivityState.STANDING,
                BedOccupancy.OUT_OF_BED,
                ObservationSource.FUSED,
            ),
        ):
            with self.subTest(gap=gap.source):
                summary = self.summarize(
                    report(
                        state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
                        state(3, 30, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
                        gap,
                        state(31, 100, ActivityState.WALKING, BedOccupancy.OUT_OF_BED),
                        events=(exit_event(3, 5),),
                    )
                )
                self.assertEqual(Decision.MONITOR, summary.decision.decision)
                self.assertEqual("confirmed_bed_exit", summary.decision.reason)

    def test_prolonged_unknown_is_monitor_but_short_unknown_is_normal(self) -> None:
        long = self.summarize(report(state(0, 8, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN)))
        short = self.summarize(report(state(0, 4, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN)))

        self.assertEqual(Decision.MONITOR, long.decision.decision)
        self.assertEqual("prolonged_unknown", long.decision.reason)
        self.assertEqual(5, long.decision.trigger_sec)
        self.assertEqual(Decision.NORMAL, short.decision.decision)


if __name__ == "__main__":
    unittest.main()
