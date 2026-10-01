"""Convert sampled video frames into typed, conservative vision features."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any

from elderly_care_agent.application.video_ports import VideoSourceFactory
from elderly_care_agent.application.video_sampling import TimestampedFrameSampler
from elderly_care_agent.application.vision_ports import PoseEstimatorFactory, PoseModelProvider
from elderly_care_agent.config import VisionSettings
from elderly_care_agent.domain.video import FrameReference
from elderly_care_agent.domain.vision import (
    BedRegion,
    BedRelation,
    NormalizedPoint,
    PersonStatus,
    PoseLandmark,
    VisionFrame,
)

logger = logging.getLogger(__name__)


class VisionFeatureExtractor:
    """Derives visible torso geometry without guessing hidden body state."""

    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12
    _LEFT_HIP = 23
    _RIGHT_HIP = 24

    def __init__(self, minimum_landmark_confidence: float = 0.5) -> None:
        if not 0 <= minimum_landmark_confidence <= 1:
            raise ValueError("minimum_landmark_confidence must be within [0, 1]")
        self._minimum_confidence = minimum_landmark_confidence

    def extract(
        self,
        reference: FrameReference,
        landmarks: tuple[PoseLandmark, ...],
        bed_region: BedRegion | None,
    ) -> VisionFrame:
        if not landmarks:
            return VisionFrame(
                reference,
                PersonStatus.UNKNOWN,
                BedRelation.UNKNOWN,
                None,
                None,
                (),
                "pose_not_detected",
            )

        visible = {
            landmark.index: landmark
            for landmark in landmarks
            if landmark.visibility >= self._minimum_confidence
            and landmark.presence >= self._minimum_confidence
            and 0 <= landmark.x <= 1
            and 0 <= landmark.y <= 1
        }
        shoulders = self._midpoint(visible, self._LEFT_SHOULDER, self._RIGHT_SHOULDER)
        hips = self._midpoint(visible, self._LEFT_HIP, self._RIGHT_HIP)
        anchor = hips or shoulders
        torso_angle = self._torso_angle(shoulders, hips)

        if anchor is None:
            relation = BedRelation.UNKNOWN
            reason = "torso_occluded"
        elif bed_region is None:
            relation = BedRelation.UNKNOWN
            reason = "bed_region_not_set"
        else:
            relation = BedRelation.INSIDE if bed_region.contains(anchor) else BedRelation.OUTSIDE
            reason = None

        return VisionFrame(
            reference, PersonStatus.DETECTED, relation, anchor, torso_angle, landmarks, reason
        )

    @staticmethod
    def _midpoint(
        visible: dict[int, PoseLandmark], left_index: int, right_index: int
    ) -> NormalizedPoint | None:
        left = visible.get(left_index)
        right = visible.get(right_index)
        if left is None or right is None:
            return None
        return NormalizedPoint((left.x + right.x) / 2, (left.y + right.y) / 2)

    @staticmethod
    def _torso_angle(
        shoulders: NormalizedPoint | None, hips: NormalizedPoint | None
    ) -> float | None:
        if shoulders is None or hips is None:
            return None
        dx = hips.x - shoulders.x
        dy = hips.y - shoulders.y
        if hypot(dx, dy) < 1e-6:
            return None
        return degrees(atan2(abs(dy), abs(dx)))


@dataclass(frozen=True, slots=True)
class VisionAnalysisReport:
    video_id: str
    source_path: str
    duration_sec: float
    source_fps: float
    width: int
    height: int
    sample_fps: float
    model_path: str
    bed_region: BedRegion | None
    sampled_frame_count: int
    detected_frame_count: int
    unknown_frame_count: int
    frames: tuple[VisionFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisionAnalysisService[FramePayload]:
    """Coordinates sampling, pose estimation, and frame-level geometry."""

    def __init__(
        self,
        source_factory: VideoSourceFactory[FramePayload],
        pose_factory: PoseEstimatorFactory[FramePayload],
        model_provider: PoseModelProvider,
        settings: VisionSettings,
        extractor: VisionFeatureExtractor | None = None,
    ) -> None:
        self._source_factory = source_factory
        self._pose_factory = pose_factory
        self._model_provider = model_provider
        self._settings = settings
        self._extractor = extractor or VisionFeatureExtractor()

    def analyze(
        self,
        path: Path,
        sample_fps: float | None = None,
        bed_region: BedRegion | None = None,
    ) -> VisionAnalysisReport:
        rate = self._settings.sample_fps if sample_fps is None else sample_fps
        source = self._source_factory.create(path)
        sampler = TimestampedFrameSampler(source, rate)
        model_path = self._model_provider.ensure_available()
        logger.info(
            "Vision analysis started | video=%s | sample_fps=%.3f | bed_region=%s",
            source.metadata.video_id,
            sampler.effective_fps,
            bed_region,
        )
        frames: list[VisionFrame] = []
        with self._pose_factory.create(
            model_path,
            self._settings.minimum_detection_confidence,
            self._settings.minimum_tracking_confidence,
        ) as estimator:
            for frame in sampler.samples():
                landmarks = estimator.estimate(frame.payload, frame.timestamp_sec)
                observation = self._extractor.extract(
                    FrameReference(frame.frame_index, frame.timestamp_sec),
                    landmarks,
                    bed_region,
                )
                frames.append(observation)
                logger.debug(
                    "Vision frame | index=%d | time=%.3f | person=%s | bed_relation=%s",
                    frame.frame_index,
                    frame.timestamp_sec,
                    observation.person_status,
                    observation.bed_relation,
                )

        detected_count = sum(frame.person_status is PersonStatus.DETECTED for frame in frames)
        report = VisionAnalysisReport(
            video_id=source.metadata.video_id,
            source_path=source.metadata.source_path,
            duration_sec=source.metadata.duration_sec,
            source_fps=source.metadata.fps,
            width=source.metadata.width,
            height=source.metadata.height,
            sample_fps=sampler.effective_fps,
            model_path=str(model_path),
            bed_region=bed_region,
            sampled_frame_count=len(frames),
            detected_frame_count=detected_count,
            unknown_frame_count=len(frames) - detected_count,
            frames=tuple(frames),
        )
        logger.info(
            "Vision analysis completed | video=%s | sampled=%d | detected=%d | unknown=%d",
            report.video_id,
            report.sampled_frame_count,
            report.detected_frame_count,
            report.unknown_frame_count,
        )
        return report
