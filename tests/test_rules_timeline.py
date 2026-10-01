import unittest
from pathlib import Path
from unittest.mock import Mock

from elderly_care_agent.application.rules import RuleBasedStateClassifier, RuleObservation
from elderly_care_agent.application.timeline import TemporalStateSmoother, TimelineAnalysisService
from elderly_care_agent.application.vision_features import VisionAnalysisReport
from elderly_care_agent.config import TemporalSettings
from elderly_care_agent.domain.enums import ActivityState, BedOccupancy
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.video import FrameReference
from elderly_care_agent.domain.vision import (
    BedRegion,
    BedRelation,
    NormalizedPoint,
    PersonStatus,
    PoseLandmark,
    VisionFrame,
)


def pose_landmark(index: int, x: float, y: float, visibility: float = 0.95) -> PoseLandmark:
    return PoseLandmark(index, x, y, 0.0, visibility, visibility)


def frame(
    time: float,
    angle: float | None,
    relation: BedRelation,
    *,
    anchor_x: float = 0.4,
    sitting_legs: bool = True,
    detected: bool = True,
    visible_legs: bool = True,
) -> VisionFrame:
    knee_y = 0.5 if sitting_legs else 0.55
    hip_y = 0.45 if sitting_legs else 0.3
    legs = (
        pose_landmark(23, anchor_x, hip_y),
        pose_landmark(25, anchor_x + 0.02, knee_y),
        pose_landmark(27, anchor_x + 0.02, 0.8),
        pose_landmark(24, anchor_x + 0.1, hip_y),
        pose_landmark(26, anchor_x + 0.1, knee_y),
        pose_landmark(28, anchor_x + 0.1, 0.8),
    )
    return VisionFrame(
        reference=FrameReference(round(time * 10), time),
        person_status=PersonStatus.DETECTED if detected else PersonStatus.UNKNOWN,
        bed_relation=relation,
        body_anchor=NormalizedPoint(anchor_x, hip_y) if detected else None,
        torso_angle_deg=angle,
        landmarks=legs if detected and visible_legs else (),
    )


def observation(time: float, activity: ActivityState, occupancy: BedOccupancy) -> RuleObservation:
    return RuleObservation(
        FrameReference(round(time * 10), time),
        activity,
        occupancy,
        0.7 if activity is not ActivityState.UNKNOWN else 0.0,
        "fixture",
        None,
        None,
    )


class RuleBasedStateClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RuleBasedStateClassifier()

    def test_sitting_and_lying_require_bed_evidence(self) -> None:
        sitting = self.classifier.classify(frame(0, 85, BedRelation.INSIDE), None, 1280, 720)
        lying = self.classifier.classify(frame(1, 15, BedRelation.INSIDE), None, 1280, 720)
        uncertain = self.classifier.classify(frame(2, 15, BedRelation.OUTSIDE), None, 1280, 720)

        self.assertEqual(
            (ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED),
            (sitting.activity, sitting.bed_occupancy),
        )
        self.assertEqual(
            (ActivityState.LYING_IN_BED, BedOccupancy.IN_BED), (lying.activity, lying.bed_occupancy)
        )
        self.assertEqual(ActivityState.UNKNOWN, uncertain.activity)
        self.assertEqual("lying_without_bed_evidence", uncertain.reason)

    def test_standing_and_walking_use_leg_geometry_and_movement(self) -> None:
        first = frame(0, 85, BedRelation.OUTSIDE, anchor_x=0.2, sitting_legs=False)
        second = frame(1, 85, BedRelation.OUTSIDE, anchor_x=0.4, sitting_legs=False)
        standing = self.classifier.classify(first, None, 1280, 720)
        walking = self.classifier.classify(second, first, 1280, 720)

        self.assertEqual(ActivityState.STANDING, standing.activity)
        self.assertEqual(BedOccupancy.OUT_OF_BED, standing.bed_occupancy)
        self.assertEqual(ActivityState.WALKING, walking.activity)
        self.assertGreater(walking.anchor_speed_per_sec, 0.08)

    def test_standing_near_bed_does_not_claim_bed_occupancy(self) -> None:
        standing = self.classifier.classify(
            frame(0, 85, BedRelation.INSIDE, sitting_legs=False), None, 1280, 720
        )

        self.assertEqual(ActivityState.STANDING, standing.activity)
        self.assertEqual(BedOccupancy.UNKNOWN, standing.bed_occupancy)

    def test_missing_pose_or_occluded_legs_stays_unknown(self) -> None:
        missing = self.classifier.classify(
            frame(0, None, BedRelation.UNKNOWN, detected=False), None, 1280, 720
        )
        occluded = self.classifier.classify(
            frame(1, 80, BedRelation.INSIDE, visible_legs=False), None, 1280, 720
        )
        transition = self.classifier.classify(frame(2, 40, BedRelation.INSIDE), None, 1280, 720)

        self.assertEqual("pose_not_detected", missing.reason)
        self.assertEqual("legs_not_visible", occluded.reason)
        self.assertEqual("transition_posture", transition.reason)
        self.assertTrue(
            all(item.activity is ActivityState.UNKNOWN for item in (missing, occluded, transition))
        )


class TemporalStateSmootherTests(unittest.TestCase):
    def test_confirmed_states_and_short_gap_cover_exact_duration(self) -> None:
        sitting = ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED
        lying = ActivityState.LYING_IN_BED, BedOccupancy.IN_BED
        brief = ActivityState.STANDING, BedOccupancy.OUT_OF_BED
        samples = (
            observation(0.0, *sitting),
            observation(0.5, *sitting),
            observation(1.0, *sitting),
            observation(1.5, *brief),
            observation(2.0, *lying),
            observation(2.5, *lying),
            observation(3.0, *lying),
            observation(3.5, *lying),
            observation(4.0, *lying),
            observation(4.5, *lying),
        )

        timeline = TemporalStateSmoother(1.5).build("clip", 5.0, samples)

        self.assertEqual(
            [ActivityState.SITTING_ON_BED, ActivityState.UNKNOWN, ActivityState.LYING_IN_BED],
            [segment.activity for segment in timeline.segments],
        )
        self.assertEqual(
            [(0.0, 1.5), (1.5, 2.0), (2.0, 5.0)],
            [(item.time_range.start_sec, item.time_range.end_sec) for item in timeline.segments],
        )
        self.assertEqual(3.0, timeline.activity_durations()[ActivityState.LYING_IN_BED])
        self.assertEqual(0.5, timeline.occupancy_durations()[BedOccupancy.UNKNOWN])
        self.assertEqual(5.0, sum(timeline.activity_durations().values()))

    def test_one_sample_never_confirms_and_missing_input_covers_video(self) -> None:
        sample = (observation(0.0, ActivityState.LYING_IN_BED, BedOccupancy.IN_BED),)
        smoother = TemporalStateSmoother(1.5)

        self.assertEqual(
            ActivityState.UNKNOWN,
            smoother.build("clip", 4.0, sample).segments[0].activity,
        )
        self.assertEqual(
            4.0,
            smoother.build("clip", 4.0, ()).activity_durations()[ActivityState.UNKNOWN],
        )

    def test_out_of_order_observations_are_rejected(self) -> None:
        state = ActivityState.STANDING, BedOccupancy.OUT_OF_BED
        samples = (observation(0.0, *state), observation(1.0, *state), observation(0.5, *state))
        with self.assertRaisesRegex(DomainValidationError, "ordered"):
            TemporalStateSmoother(1.5).build("clip", 2.0, samples)


class TimelineAnalysisServiceTests(unittest.TestCase):
    def test_vision_is_run_once_and_report_sums_to_video_duration(self) -> None:
        vision_service = Mock()
        vision_service.analyze.return_value = VisionAnalysisReport(
            video_id="clip.mp4",
            source_path="clip.mp4",
            duration_sec=2.0,
            source_fps=10.0,
            width=1280,
            height=720,
            sample_fps=2.0,
            model_path="pose.task",
            bed_region=BedRegion(0.1, 0.2, 0.9, 0.8),
            sampled_frame_count=4,
            detected_frame_count=4,
            unknown_frame_count=0,
            frames=tuple(frame(time, 85, BedRelation.INSIDE) for time in (0, 0.5, 1, 1.5)),
        )
        service = TimelineAnalysisService(vision_service, TemporalSettings())

        report = service.analyze(Path("clip.mp4"), 2.0, BedRegion(0.1, 0.2, 0.9, 0.8))

        self.assertEqual(2.0, report.activity_durations_sec["sitting_on_bed"])
        self.assertEqual(2.0, sum(report.activity_durations_sec.values()))
        self.assertEqual(1, len(report.segments))
        vision_service.analyze.assert_called_once()


if __name__ == "__main__":
    unittest.main()
