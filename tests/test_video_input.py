import tempfile
import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from elderly_care_agent.application.video_ports import VideoSource
from elderly_care_agent.application.video_sampling import (
    TimestampedFrameSampler,
    VideoSamplingService,
)
from elderly_care_agent.domain.exceptions import DomainValidationError, VideoInputError
from elderly_care_agent.domain.video import TimestampedFrame, VideoMetadata
from elderly_care_agent.infrastructure.opencv_video_source import (
    OpenCvVideoSource,
    OpenCvVideoSourceFactory,
)


class FakeVideoSource(VideoSource[str]):
    def __init__(self) -> None:
        self._metadata = VideoMetadata("fake.mp4", "fake.mp4", 10.0, 20, 64, 48)
        self.requested_indices: tuple[int, ...] = ()

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def read_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[TimestampedFrame[str]]:
        self.requested_indices = tuple(frame_indices)
        for frame_index in frame_indices:
            yield TimestampedFrame(frame_index, frame_index / self.metadata.fps, "payload")


class TimestampedFrameSamplerTests(unittest.TestCase):
    def test_selects_exact_indices_and_source_timestamps(self) -> None:
        source = FakeVideoSource()
        sampler = TimestampedFrameSampler(source, requested_fps=2.0)

        samples = tuple(sampler.samples())

        self.assertEqual((0, 5, 10, 15), source.requested_indices)
        self.assertEqual([0.0, 0.5, 1.0, 1.5], [sample.timestamp_sec for sample in samples])
        self.assertEqual(2.0, sampler.effective_fps)

    def test_caps_sampling_at_source_frame_rate(self) -> None:
        sampler = TimestampedFrameSampler(FakeVideoSource(), requested_fps=30.0)

        self.assertEqual(tuple(range(20)), sampler.frame_indices())
        self.assertEqual(10.0, sampler.effective_fps)

    def test_rejects_non_positive_sample_rate(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "requested_fps"):
            TimestampedFrameSampler(FakeVideoSource(), requested_fps=0)


class OpenCvVideoSourceTests(unittest.TestCase):
    def test_synthetic_video_returns_metadata_count_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "synthetic.avi"
            self._write_synthetic_video(video_path)

            source = OpenCvVideoSource(video_path)
            report = VideoSamplingService(OpenCvVideoSourceFactory()).inspect(
                video_path,
                sample_fps=2.0,
            )
            decoded = tuple(source.read_frames((0, 5, 10, 15)))

            self.assertEqual(10.0, source.metadata.fps)
            self.assertEqual(20, source.metadata.frame_count)
            self.assertEqual((64, 48), (source.metadata.width, source.metadata.height))
            self.assertAlmostEqual(2.0, source.metadata.duration_sec)
            self.assertEqual(4, report.sampled_frame_count)
            self.assertEqual([0.0, 0.5, 1.0, 1.5], [item.timestamp_sec for item in report.samples])
            self.assertEqual((48, 64, 3), decoded[0].payload.shape)

    def test_missing_video_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(VideoInputError, "does not exist"):
                OpenCvVideoSource(root / "missing.mp4")

    def test_invalid_indices_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "synthetic.avi"
            self._write_synthetic_video(video_path)
            source = OpenCvVideoSource(video_path)
            with self.assertRaisesRegex(VideoInputError, "strictly increasing"):
                tuple(source.read_frames((5, 5)))
            with self.assertRaisesRegex(VideoInputError, "outside"):
                tuple(source.read_frames((20,)))

    def _write_synthetic_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened(), "test environment cannot create MJPG video")
        try:
            for frame_index in range(20):
                frame = np.full((48, 64, 3), frame_index * 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
