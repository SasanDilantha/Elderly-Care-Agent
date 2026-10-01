import unittest
from pathlib import Path
from unittest.mock import Mock

from elderly_care_agent.application.bed_events import (
    BedEventAnalysisService,
    BedTransitionDetector,
    ConservativeTimelineFusion,
)
from elderly_care_agent.application.timeline import TimelineAnalysisReport
from elderly_care_agent.application.vlm_review import VlmReviewReport
from elderly_care_agent.config import TemporalSettings
from elderly_care_agent.domain.enums import (
    ActivityState,
    BedOccupancy,
    Decision,
    EventType,
    ObservationSource,
)
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import StateSegment, StateTimeline, TimeRange
from elderly_care_agent.domain.video import FrameReference
from elderly_care_agent.domain.vlm import ReviewStatus, VlmReview


def state(
    start: float,
    end: float,
    activity: ActivityState,
    occupancy: BedOccupancy,
    source: ObservationSource = ObservationSource.RULES,
) -> StateSegment:
    confidence = 0.0 if activity is ActivityState.UNKNOWN else 0.7
    return StateSegment(TimeRange(start, end), activity, occupancy, confidence, source)


def review(
    start: float,
    end: float,
    activity: ActivityState,
    occupancy: BedOccupancy,
    confidence: float = 0.8,
) -> VlmReview:
    return VlmReview(
        TimeRange(start, end),
        ReviewStatus.PROPOSED,
        activity,
        occupancy,
        confidence,
        "Visible posture is consistent.",
        (FrameReference(3, 1.5),),
    )


def timeline(*segments: StateSegment) -> StateTimeline:
    return StateTimeline("clip.mp4", segments[-1].time_range.end_sec, segments)


class ConservativeTimelineFusionTests(unittest.TestCase):
    def test_fills_only_enclosed_same_occupancy_gap(self) -> None:
        rules = timeline(
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 5, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN),
            state(5, 8, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),
        )

        fused = ConservativeTimelineFusion().fuse(
            rules, (review(3, 5, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),)
        )

        self.assertEqual(ActivityState.LYING_IN_BED, fused.segments[1].activity)
        self.assertEqual(ObservationSource.FUSED, fused.segments[1].source)
        self.assertEqual(ActivityState.UNKNOWN, rules.segments[1].activity)
        self.assertEqual(8, sum(fused.activity_durations().values()))

    def test_conflicting_boundary_abstention_and_low_confidence_do_not_fill(self) -> None:
        rules = timeline(
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 5, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN),
            state(5, 8, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
        )
        proposed = review(3, 5, ActivityState.STANDING, BedOccupancy.OUT_OF_BED)
        abstained = VlmReview(
            proposed.time_range,
            ReviewStatus.ABSTAINED,
            ActivityState.UNKNOWN,
            BedOccupancy.UNKNOWN,
            0.0,
            "ambiguous",
            proposed.context_frames,
        )

        for candidate in (proposed, abstained):
            with self.subTest(candidate=candidate.status):
                self.assertEqual(
                    ActivityState.UNKNOWN,
                    ConservativeTimelineFusion().fuse(rules, (candidate,)).segments[1].activity,
                )
        same_bed = timeline(
            rules.segments[0],
            rules.segments[1],
            state(5, 8, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),
        )
        low_confidence = review(3, 5, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED, 0.4)
        self.assertEqual(
            ActivityState.UNKNOWN,
            ConservativeTimelineFusion().fuse(same_bed, (low_confidence,)).segments[1].activity,
        )

    def test_review_must_target_unique_unknown_rule_interval(self) -> None:
        rules = timeline(state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED))
        candidate = review(0, 3, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED)

        with self.assertRaisesRegex(DomainValidationError, "rule-unknown"):
            ConservativeTimelineFusion().fuse(rules, (candidate,))
        with self.assertRaisesRegex(DomainValidationError, "unique"):
            ConservativeTimelineFusion().fuse(rules, (candidate, candidate))

    def test_unknown_at_video_edge_is_not_filled(self) -> None:
        rules = timeline(
            state(0, 2, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN),
            state(2, 5, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),
        )

        fused = ConservativeTimelineFusion().fuse(
            rules, (review(0, 2, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),)
        )

        self.assertEqual(ActivityState.UNKNOWN, fused.segments[0].activity)


class BedTransitionDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = BedTransitionDetector(TemporalSettings())

    def test_confirms_direct_exit_and_return_after_two_seconds(self) -> None:
        states = timeline(
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 6, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
            state(6, 10, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),
        )

        events = self.detector.detect(states)

        self.assertEqual(
            (EventType.BED_EXIT, EventType.BED_RETURN), tuple(e.event_type for e in events)
        )
        self.assertEqual(((3, 5), (6, 8)), tuple((e.start_sec, e.confirmed_sec) for e in events))
        self.assertEqual((Decision.MONITOR, Decision.NORMAL), tuple(e.decision for e in events))
        self.assertEqual((0.7, 0.7), tuple(e.confidence for e in events))

    def test_unknown_or_fused_evidence_never_creates_boundary_event(self) -> None:
        base = state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED)
        unknown = state(3, 5, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN)
        outside = state(5, 8, ActivityState.STANDING, BedOccupancy.OUT_OF_BED)
        fused = state(
            3, 6, ActivityState.STANDING, BedOccupancy.OUT_OF_BED, ObservationSource.FUSED
        )

        self.assertEqual((), self.detector.detect(timeline(base, unknown, outside)))
        self.assertEqual((), self.detector.detect(timeline(base, fused)))

    def test_short_current_or_previous_state_does_not_confirm(self) -> None:
        short_current = timeline(
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 4, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
        )
        short_previous = timeline(
            state(0, 1, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(1, 4, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
        )

        self.assertEqual((), self.detector.detect(short_current))
        self.assertEqual((), self.detector.detect(short_previous))


class BedEventAnalysisServiceTests(unittest.TestCase):
    def test_rules_only_does_not_call_vlm(self) -> None:
        segments = (
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 6, ActivityState.STANDING, BedOccupancy.OUT_OF_BED),
        )
        timeline_service = Mock()
        timeline_service.analyze.return_value = TimelineAnalysisReport(
            "clip.mp4", "clip.mp4", 6, 2, None, 12, (), segments, {}, {}
        )
        reviewer = Mock()
        service = BedEventAnalysisService(timeline_service, TemporalSettings(), reviewer)

        result = service.analyze(Path("clip.mp4"))

        self.assertEqual(1, len(result.events))
        self.assertEqual(1, result.to_dict()["event_count"])
        self.assertFalse(result.vlm_enabled)
        reviewer.review.assert_not_called()
        timeline_service.analyze.assert_called_once()

    def test_vlm_mode_uses_review_rule_timeline_without_second_rule_run(self) -> None:
        segments = (
            state(0, 3, ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            state(3, 5, ActivityState.UNKNOWN, BedOccupancy.UNKNOWN),
            state(5, 8, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),
        )
        reviewer = Mock()
        reviewer.review.return_value = VlmReviewReport(
            "clip.mp4",
            "clip.mp4",
            8,
            "qwen3-vl:4b-instruct",
            1,
            1,
            0,
            segments,
            (review(3, 5, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),),
        )
        timeline_service = Mock()
        service = BedEventAnalysisService(timeline_service, TemporalSettings(), reviewer)

        result = service.analyze(Path("clip.mp4"), with_vlm=True, max_segments=1)

        self.assertEqual(ActivityState.LYING_IN_BED, result.fused_segments[1].activity)
        self.assertEqual(1, result.to_dict()["promoted_vlm_segment_count"])
        self.assertEqual((), result.events)
        reviewer.review.assert_called_once_with(Path("clip.mp4"), None, None, 1)
        timeline_service.analyze.assert_not_called()

    def test_options_fail_without_vlm_service_or_vlm_mode(self) -> None:
        service = BedEventAnalysisService(Mock(), TemporalSettings())

        with self.assertRaisesRegex(DomainValidationError, "with_vlm"):
            service.analyze(Path("clip.mp4"), max_segments=1)
        with self.assertRaisesRegex(DomainValidationError, "unavailable"):
            service.analyze(Path("clip.mp4"), with_vlm=True)


if __name__ == "__main__":
    unittest.main()
