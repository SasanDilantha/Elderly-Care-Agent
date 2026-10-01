"""Deterministic timestamped-frame sampling use case."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from elderly_care_agent.application.video_ports import VideoSource, VideoSourceFactory
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.video import FrameReference, TimestampedFrame, VideoMetadata

logger = logging.getLogger(__name__)


class TimestampedFrameSampler[FramePayload]:
    """Selects reproducible source-frame indices at a requested rate."""

    def __init__(self, source: VideoSource[FramePayload], requested_fps: float) -> None:
        if not isfinite(requested_fps) or requested_fps <= 0:
            raise DomainValidationError("requested_fps must be finite and greater than zero")
        self._source = source
        self._requested_fps = requested_fps

    @property
    def effective_fps(self) -> float:
        return min(self._requested_fps, self._source.metadata.fps)

    @property
    def requested_fps(self) -> float:
        return self._requested_fps

    def frame_indices(self) -> tuple[int, ...]:
        """Return ordered, unique indices without exceeding source FPS."""

        metadata = self._source.metadata
        if self._requested_fps >= metadata.fps:
            return tuple(range(metadata.frame_count))

        step = metadata.fps / self._requested_fps
        indices: list[int] = []
        sample_number = 0
        while True:
            frame_index = int(sample_number * step + 0.5)
            if frame_index >= metadata.frame_count:
                break
            if not indices or frame_index != indices[-1]:
                indices.append(frame_index)
            sample_number += 1
        return tuple(indices)

    def samples(self) -> Iterator[TimestampedFrame[FramePayload]]:
        """Decode and yield only selected frames."""

        indices = self.frame_indices()
        logger.info(
            "Frame sampling started | requested_fps=%.3f | effective_fps=%.3f | samples=%d",
            self._requested_fps,
            self.effective_fps,
            len(indices),
        )
        yield from self._source.read_frames(indices)
        logger.info("Frame sampling completed | samples=%d", len(indices))


@dataclass(frozen=True, slots=True)
class VideoSamplingReport:
    """JSON-safe result produced by the video sampling service."""

    video_id: str
    source_path: str
    source_fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    requested_sample_fps: float
    effective_sample_fps: float
    sampled_frame_count: int
    samples: tuple[FrameReference, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoSamplingService[FramePayload]:
    """Coordinates source creation, sampling, and a UI-neutral report."""

    def __init__(self, source_factory: VideoSourceFactory[FramePayload]) -> None:
        self._source_factory = source_factory

    def inspect(self, path: Path, sample_fps: float) -> VideoSamplingReport:
        logger.info("Video inspection started | path=%s | sample_fps=%.3f", path, sample_fps)
        source = self._source_factory.create(path)
        sampler = TimestampedFrameSampler(source, sample_fps)
        samples = tuple(
            FrameReference(frame.frame_index, frame.timestamp_sec) for frame in sampler.samples()
        )
        report = self._build_report(source.metadata, sampler, samples)
        logger.info(
            "Video inspection completed | video=%s | frames=%d | sampled=%d | duration=%.3f",
            report.video_id,
            report.frame_count,
            report.sampled_frame_count,
            report.duration_sec,
        )
        return report

    @staticmethod
    def _build_report(
        metadata: VideoMetadata,
        sampler: TimestampedFrameSampler[FramePayload],
        samples: tuple[FrameReference, ...],
    ) -> VideoSamplingReport:
        return VideoSamplingReport(
            video_id=metadata.video_id,
            source_path=metadata.source_path,
            source_fps=metadata.fps,
            frame_count=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
            duration_sec=metadata.duration_sec,
            requested_sample_fps=sampler.requested_fps,
            effective_sample_fps=sampler.effective_fps,
            sampled_frame_count=len(samples),
            samples=samples,
        )
