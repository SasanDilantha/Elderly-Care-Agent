import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import Mock

from elderly_care_agent.application.video_ports import VideoSource, VideoSourceFactory
from elderly_care_agent.application.vision_features import (
    VisionAnalysisService,
    VisionFeatureExtractor,
)
from elderly_care_agent.application.vision_ports import (
    PoseEstimator,
    PoseEstimatorFactory,
    PoseModelProvider,
)
from elderly_care_agent.config import VisionSettings
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.video import FrameReference, TimestampedFrame, VideoMetadata
from elderly_care_agent.domain.vision import (
    BedRegion,
    BedRelation,
    PersonStatus,
    PoseLandmark,
)


def landmark(index: int, x: float, y: float, confidence: float = 0.95) -> PoseLandmark:
    return PoseLandmark(index, x, y, 0.0, confidence, confidence)


class VisionFeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = VisionFeatureExtractor()
        self.reference = FrameReference(5, 0.5)
        self.bed = BedRegion(0.2, 0.3, 0.8, 0.9)

    def test_visible_pose_produces_torso_geometry_and_inside_relation(self) -> None:
        pose = (
            landmark(11, 0.4, 0.4),
            landmark(12, 0.6, 0.4),
            landmark(23, 0.4, 0.7),
            landmark(24, 0.6, 0.7),
        )

        frame = self.extractor.extract(self.reference, pose, self.bed)

        self.assertEqual(PersonStatus.DETECTED, frame.person_status)
        self.assertEqual(BedRelation.INSIDE, frame.bed_relation)
        self.assertAlmostEqual(0.5, frame.body_anchor.x)
        self.assertAlmostEqual(0.7, frame.body_anchor.y)
        self.assertAlmostEqual(90.0, frame.torso_angle_deg)
        self.assertIsNone(frame.unknown_reason)

    def test_outside_region_uses_hip_anchor(self) -> None:
        pose = (landmark(23, 0.85, 0.5), landmark(24, 0.95, 0.5))

        frame = self.extractor.extract(self.reference, pose, self.bed)

        self.assertEqual(BedRelation.OUTSIDE, frame.bed_relation)
        self.assertIsNone(frame.torso_angle_deg)

    def test_no_pose_and_occluded_torso_are_unknown(self) -> None:
        missing = self.extractor.extract(self.reference, (), self.bed)
        occluded = self.extractor.extract(
            self.reference,
            (landmark(23, 0.5, 0.6, 0.1), landmark(24, 0.6, 0.6, 0.1)),
            self.bed,
        )

        self.assertEqual(PersonStatus.UNKNOWN, missing.person_status)
        self.assertEqual(BedRelation.UNKNOWN, missing.bed_relation)
        self.assertEqual("pose_not_detected", missing.unknown_reason)
        self.assertEqual(PersonStatus.DETECTED, occluded.person_status)
        self.assertEqual(BedRelation.UNKNOWN, occluded.bed_relation)
        self.assertEqual("torso_occluded", occluded.unknown_reason)

    def test_missing_bed_region_stays_unknown_with_visible_pose(self) -> None:
        frame = self.extractor.extract(
            self.reference,
            (landmark(23, 0.4, 0.7), landmark(24, 0.6, 0.7)),
            None,
        )

        self.assertEqual(PersonStatus.DETECTED, frame.person_status)
        self.assertEqual(BedRelation.UNKNOWN, frame.bed_relation)
        self.assertEqual("bed_region_not_set", frame.unknown_reason)

    def test_invalid_region_is_rejected(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "positive width"):
            BedRegion.parse("0.4,0.2,0.4,0.8")
        with self.assertRaisesRegex(DomainValidationError, "within"):
            BedRegion.parse("0,0,1.2,1")


class FakeSource(VideoSource[str]):
    metadata = VideoMetadata("clip.mp4", "clip.mp4", 10.0, 10, 64, 48)

    def read_frames(self, frame_indices: Sequence[int]) -> Iterator[TimestampedFrame[str]]:
        for index in frame_indices:
            yield TimestampedFrame(index, index / 10, f"frame-{index}")


class FakeSourceFactory(VideoSourceFactory[str]):
    def create(self, path: Path) -> FakeSource:
        return FakeSource()


class FakeEstimator(PoseEstimator[str]):
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.closed = False

    def estimate(self, frame: str, timestamp_sec: float) -> tuple[PoseLandmark, ...]:
        self.calls.append((frame, timestamp_sec))
        if frame == "frame-0":
            return (landmark(23, 0.4, 0.6), landmark(24, 0.6, 0.6))
        return ()

    def close(self) -> None:
        self.closed = True


class FakeEstimatorFactory(PoseEstimatorFactory[str]):
    def __init__(self, estimator: FakeEstimator) -> None:
        self.estimator = estimator

    def create(
        self, model_path: Path, detection_confidence: float, tracking_confidence: float
    ) -> FakeEstimator:
        return self.estimator


class VisionAnalysisServiceTests(unittest.TestCase):
    def test_samples_frames_and_closes_pose_backend(self) -> None:
        estimator = FakeEstimator()
        model_provider = Mock(spec=PoseModelProvider)
        model_provider.ensure_available.return_value = Path("model.task")
        service = VisionAnalysisService(
            FakeSourceFactory(),
            FakeEstimatorFactory(estimator),
            model_provider,
            VisionSettings(sample_fps=2.0),
        )

        report = service.analyze(Path("clip.mp4"), bed_region=BedRegion(0.2, 0.2, 0.8, 0.8))

        self.assertEqual(2, report.sampled_frame_count)
        self.assertEqual(1, report.detected_frame_count)
        self.assertEqual(1, report.unknown_frame_count)
        self.assertEqual(
            [(0, 0.0), (5, 0.5)],
            [
                (frame.reference.frame_index, frame.reference.timestamp_sec)
                for frame in report.frames
            ],
        )
        self.assertEqual(BedRelation.INSIDE, report.frames[0].bed_relation)
        self.assertEqual(BedRelation.UNKNOWN, report.frames[1].bed_relation)
        self.assertEqual([("frame-0", 0.0), ("frame-5", 0.5)], estimator.calls)
        self.assertTrue(estimator.closed)
        model_provider.ensure_available.assert_called_once()


if __name__ == "__main__":
    unittest.main()
