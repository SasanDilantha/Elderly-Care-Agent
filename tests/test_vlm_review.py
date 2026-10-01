import json
import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from elderly_care_agent.application.timeline import TimelineAnalysisReport
from elderly_care_agent.application.video_ports import VideoSource, VideoSourceFactory
from elderly_care_agent.application.vlm_ports import ContextFrameEncoder, VisionLanguageModel
from elderly_care_agent.application.vlm_review import (
    ContextFrameSampler,
    VlmResponseParser,
    VlmReviewService,
)
from elderly_care_agent.config import VlmSettings
from elderly_care_agent.domain.enums import ActivityState, BedOccupancy, ObservationSource
from elderly_care_agent.domain.models import StateSegment, TimeRange
from elderly_care_agent.domain.video import FrameReference, TimestampedFrame, VideoMetadata
from elderly_care_agent.domain.vlm import ReviewStatus
from elderly_care_agent.infrastructure.vlm_adapters import (
    OllamaVisionLanguageModel,
    OpenCvJpegEncoder,
)


def segment(start: float, end: float, activity: ActivityState) -> StateSegment:
    occupancy = BedOccupancy.UNKNOWN if activity is ActivityState.UNKNOWN else BedOccupancy.IN_BED
    return StateSegment(TimeRange(start, end), activity, occupancy, 0.0, ObservationSource.RULES)


def response(activity: str, occupancy: str, confidence: float = 0.8) -> str:
    return json.dumps(
        {
            "activity": activity,
            "bed_occupancy": occupancy,
            "confidence": confidence,
            "reason": "Visible person beside the bed.",
        }
    )


class FakeSource(VideoSource[str]):
    metadata = VideoMetadata("clip.mp4", "clip.mp4", 2.0, 20, 64, 48)

    def read_frames(self, indices: Sequence[int]) -> Iterator[TimestampedFrame[str]]:
        for index in indices:
            yield TimestampedFrame(index, index / 2, f"frame-{index}")


class FakeSourceFactory(VideoSourceFactory[str]):
    def __init__(self) -> None:
        self.calls = 0

    def create(self, path: Path) -> FakeSource:
        self.calls += 1
        return FakeSource()


class FakeEncoder(ContextFrameEncoder[str]):
    def encode(self, frame: str) -> bytes:
        return frame.encode()


class ContextFrameSamplerTests(unittest.TestCase):
    def test_clamps_deduplicates_and_keeps_video_order(self) -> None:
        frames = ContextFrameSampler().sample(
            FakeSource(),
            segment(0.0, 4.0, ActivityState.UNKNOWN),
            (-4.0, -2.0, 0.0, 2.0, 4.0),
        )

        self.assertEqual((0, 4, 8, 12), tuple(frame.frame_index for frame in frames))
        self.assertEqual((0.0, 2.0, 4.0, 6.0), tuple(frame.timestamp_sec for frame in frames))


class VlmResponseParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = VlmResponseParser(0.6)
        self.segment = segment(1.0, 2.0, ActivityState.UNKNOWN)
        self.references = (FrameReference(1, 0.5), FrameReference(3, 1.5))

    def test_valid_model_result_becomes_separate_proposal(self) -> None:
        review = self.parser.parse(
            response("sitting_on_bed", "in_bed"), self.segment, self.references
        )

        self.assertEqual(ReviewStatus.PROPOSED, review.status)
        self.assertEqual(ActivityState.SITTING_ON_BED, review.activity)
        self.assertEqual(ObservationSource.VLM, review.source)
        self.assertEqual(self.references, review.context_frames)

    def test_malformed_conflicting_and_low_confidence_results_abstain(self) -> None:
        candidates = (
            "not json",
            response("lying_in_bed", "out_of_bed"),
            response("walking", "out_of_bed", confidence=0.2),
            response("unknown", "unknown"),
            '{"activity":"walking","bed_occupancy":"out_of_bed","confidence":true,"reason":"x"}',
        )

        reviews = [self.parser.parse(item, self.segment, self.references) for item in candidates]

        self.assertTrue(all(review.status is ReviewStatus.ABSTAINED for review in reviews))
        self.assertTrue(all(review.activity is ActivityState.UNKNOWN for review in reviews))
        self.assertEqual("low_model_confidence", reviews[2].reason)


class VlmReviewServiceTests(unittest.TestCase):
    def _timeline(self, *segments: StateSegment) -> TimelineAnalysisReport:
        return TimelineAnalysisReport(
            video_id="clip.mp4",
            source_path="clip.mp4",
            duration_sec=10.0,
            sample_fps=2.0,
            bed_region=None,
            sampled_frame_count=20,
            observations=(),
            segments=segments,
            activity_durations_sec={},
            occupancy_durations_sec={},
        )

    def test_only_unknown_interval_is_sent_with_ordered_context(self) -> None:
        timeline_service = Mock()
        timeline_service.analyze.return_value = self._timeline(
            segment(0, 4, ActivityState.UNKNOWN),
            segment(4, 10, ActivityState.LYING_IN_BED),
        )
        source_factory = FakeSourceFactory()
        model = Mock(spec=VisionLanguageModel)
        model.complete.return_value = response("standing", "out_of_bed")
        service = VlmReviewService(
            timeline_service, source_factory, FakeEncoder(), model, VlmSettings()
        )

        report = service.review(Path("clip.mp4"), max_segments=1)

        self.assertEqual(1, report.unknown_segment_count)
        self.assertEqual(1, report.reviewed_segment_count)
        self.assertEqual(ReviewStatus.PROPOSED, report.reviews[0].status)
        self.assertEqual(0, report.skipped_segment_count)
        self.assertEqual(1, source_factory.calls)
        prompt, images, schema = model.complete.call_args.args
        self.assertIn("0.00s to 4.00s", prompt)
        self.assertEqual([b"frame-0", b"frame-4", b"frame-8"], images)
        self.assertEqual("object", schema["type"])
        self.assertEqual(ActivityState.UNKNOWN, report.rule_segments[0].activity)

    def test_no_unknown_interval_skips_model_and_video_decode(self) -> None:
        timeline_service = Mock()
        timeline_service.analyze.return_value = self._timeline(
            segment(0, 10, ActivityState.LYING_IN_BED)
        )
        source_factory = FakeSourceFactory()
        model = Mock(spec=VisionLanguageModel)
        service = VlmReviewService(
            timeline_service, source_factory, FakeEncoder(), model, VlmSettings()
        )

        report = service.review(Path("clip.mp4"))

        self.assertEqual(0, report.reviewed_segment_count)
        self.assertEqual(0, source_factory.calls)
        model.complete.assert_not_called()

    def test_review_limit_reports_skipped_intervals(self) -> None:
        timeline_service = Mock()
        timeline_service.analyze.return_value = self._timeline(
            segment(0, 4, ActivityState.UNKNOWN),
            segment(4, 6, ActivityState.LYING_IN_BED),
            segment(6, 10, ActivityState.UNKNOWN),
        )
        model = Mock(spec=VisionLanguageModel)
        model.complete.return_value = response("standing", "out_of_bed")
        service = VlmReviewService(
            timeline_service, FakeSourceFactory(), FakeEncoder(), model, VlmSettings()
        )

        report = service.review(Path("clip.mp4"), max_segments=1)

        self.assertEqual(
            (2, 1, 1),
            (
                report.unknown_segment_count,
                report.reviewed_segment_count,
                report.skipped_segment_count,
            ),
        )
        model.complete.assert_called_once()


class AdapterTests(unittest.TestCase):
    def test_jpeg_encoder_resizes_in_memory(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        encoded = OpenCvJpegEncoder().encode(image)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual((288, 512, 3), decoded.shape)

    def test_ollama_chat_contract_uses_structured_multimodal_request(self) -> None:
        client = Mock()
        client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))
        settings = VlmSettings()
        adapter = OllamaVisionLanguageModel(settings, client)

        content = adapter.complete("describe", [b"jpeg-1", b"jpeg-2"], {"type": "object"})

        self.assertEqual('{"ok":true}', content)
        client.chat.assert_called_once_with(
            model="qwen3-vl:4b-instruct",
            messages=[{"role": "user", "content": "describe", "images": [b"jpeg-1", b"jpeg-2"]}],
            format={"type": "object"},
            options={"temperature": 0.0, "num_ctx": 4096},
            stream=False,
        )


if __name__ == "__main__":
    unittest.main()
