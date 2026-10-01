"""Review rule-abstained intervals using ordered local video context."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from elderly_care_agent.application.timeline import TimelineAnalysisService
from elderly_care_agent.application.video_ports import VideoSource, VideoSourceFactory
from elderly_care_agent.application.vlm_ports import ContextFrameEncoder, VisionLanguageModel
from elderly_care_agent.config import VlmSettings
from elderly_care_agent.domain.enums import ActivityState, BedOccupancy
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import StateSegment
from elderly_care_agent.domain.video import FrameReference, TimestampedFrame
from elderly_care_agent.domain.vision import BedRegion
from elderly_care_agent.domain.vlm import ReviewStatus, VlmReview

logger = logging.getLogger(__name__)


class ContextFrameSampler[FramePayload]:
    """Selects unique source frames around an interval midpoint."""

    def sample(
        self,
        source: VideoSource[FramePayload],
        segment: StateSegment,
        offsets_sec: Sequence[float],
    ) -> tuple[TimestampedFrame[FramePayload], ...]:
        metadata = source.metadata
        midpoint = (segment.time_range.start_sec + segment.time_range.end_sec) / 2
        last_time = (metadata.frame_count - 1) / metadata.fps
        indices = sorted(
            {
                min(
                    metadata.frame_count - 1,
                    int(min(max(midpoint + offset, 0.0), last_time) * metadata.fps + 0.5),
                )
                for offset in offsets_sec
            }
        )
        return tuple(source.read_frames(indices))


class VlmResponseParser:
    """Treats model text as untrusted and abstains on invalid claims."""

    SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "activity": {"type": "string", "enum": [item.value for item in ActivityState]},
            "bed_occupancy": {"type": "string", "enum": [item.value for item in BedOccupancy]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["activity", "bed_occupancy", "confidence", "reason"],
        "additionalProperties": False,
    }

    def __init__(self, minimum_proposal_confidence: float) -> None:
        self._minimum_confidence = minimum_proposal_confidence

    def parse(
        self,
        content: str,
        segment: StateSegment,
        references: tuple[FrameReference, ...],
    ) -> VlmReview:
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                return self._abstain(segment, references, "invalid_model_response")
            if set(payload) != {"activity", "bed_occupancy", "confidence", "reason"}:
                return self._abstain(segment, references, "invalid_model_response")
            activity = ActivityState(payload["activity"])
            occupancy = BedOccupancy(payload["bed_occupancy"])
            confidence = payload["confidence"]
            reason = payload["reason"]
        except (ValueError, TypeError, KeyError):
            return self._abstain(segment, references, "invalid_model_response")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 500
            or not self._valid_pair(activity, occupancy)
        ):
            return self._abstain(segment, references, "invalid_model_response")
        if activity is ActivityState.UNKNOWN:
            return self._abstain(segment, references, "model_abstained")
        if confidence < self._minimum_confidence:
            return self._abstain(segment, references, "low_model_confidence")
        return VlmReview(
            segment.time_range,
            ReviewStatus.PROPOSED,
            activity,
            occupancy,
            float(confidence),
            reason.strip(),
            references,
        )

    @staticmethod
    def _valid_pair(activity: ActivityState, occupancy: BedOccupancy) -> bool:
        if activity is ActivityState.UNKNOWN:
            return occupancy is BedOccupancy.UNKNOWN
        if activity in {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}:
            return occupancy is BedOccupancy.IN_BED
        return occupancy is BedOccupancy.OUT_OF_BED

    @staticmethod
    def _abstain(
        segment: StateSegment, references: tuple[FrameReference, ...], reason: str
    ) -> VlmReview:
        return VlmReview(
            segment.time_range,
            ReviewStatus.ABSTAINED,
            ActivityState.UNKNOWN,
            BedOccupancy.UNKNOWN,
            0.0,
            reason,
            references,
        )


@dataclass(frozen=True, slots=True)
class VlmReviewReport:
    video_id: str
    source_path: str
    duration_sec: float
    model: str
    unknown_segment_count: int
    reviewed_segment_count: int
    skipped_segment_count: int
    rule_segments: tuple[StateSegment, ...]
    reviews: tuple[VlmReview, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VlmReviewService[FramePayload]:
    """Calls the local model only for bounded UNKNOWN timeline segments."""

    def __init__(
        self,
        timeline_service: TimelineAnalysisService[FramePayload],
        source_factory: VideoSourceFactory[FramePayload],
        encoder: ContextFrameEncoder[FramePayload],
        model: VisionLanguageModel,
        settings: VlmSettings,
        sampler: ContextFrameSampler[FramePayload] | None = None,
    ) -> None:
        self._timeline_service = timeline_service
        self._source_factory = source_factory
        self._encoder = encoder
        self._model = model
        self._settings = settings
        self._sampler = sampler or ContextFrameSampler()
        self._parser = VlmResponseParser(settings.minimum_proposal_confidence)

    def review(
        self,
        path: Path,
        sample_fps: float | None = None,
        bed_region: BedRegion | None = None,
        max_segments: int | None = None,
    ) -> VlmReviewReport:
        limit = self._settings.max_reviews_per_run if max_segments is None else max_segments
        if limit <= 0:
            raise DomainValidationError("max_segments must be greater than zero")
        timeline = self._timeline_service.analyze(path, sample_fps, bed_region)
        unknown_segments = tuple(
            segment for segment in timeline.segments if segment.activity is ActivityState.UNKNOWN
        )
        selected = unknown_segments[:limit]
        reviews: list[VlmReview] = []
        if selected:
            source = self._source_factory.create(path)
            for segment in selected:
                frames = self._sampler.sample(source, segment, self._settings.context_offsets_sec)
                references = tuple(
                    FrameReference(frame.frame_index, frame.timestamp_sec) for frame in frames
                )
                prompt = self._prompt(segment, references)
                logger.info(
                    "VLM review started | interval=%.3f-%.3f | context_frames=%d | model=%s",
                    segment.time_range.start_sec,
                    segment.time_range.end_sec,
                    len(frames),
                    self._settings.model,
                )
                content = self._model.complete(
                    prompt,
                    [self._encoder.encode(frame.payload) for frame in frames],
                    self._parser.SCHEMA,
                )
                review = self._parser.parse(content, segment, references)
                reviews.append(review)
                logger.info(
                    "VLM review completed | interval=%.3f-%.3f | status=%s | activity=%s",
                    segment.time_range.start_sec,
                    segment.time_range.end_sec,
                    review.status,
                    review.activity,
                )

        return VlmReviewReport(
            video_id=timeline.video_id,
            source_path=timeline.source_path,
            duration_sec=timeline.duration_sec,
            model=self._settings.model,
            unknown_segment_count=len(unknown_segments),
            reviewed_segment_count=len(reviews),
            skipped_segment_count=len(unknown_segments) - len(reviews),
            rule_segments=timeline.segments,
            reviews=tuple(reviews),
        )

    @staticmethod
    def _prompt(segment: StateSegment, references: tuple[FrameReference, ...]) -> str:
        times = ", ".join(f"{frame.timestamp_sec:.2f}s" for frame in references)
        return (
            "You are reviewing a fixed-camera elderly-care video. "
            f"The rule system could not classify the interval "
            f"{segment.time_range.start_sec:.2f}s to {segment.time_range.end_sec:.2f}s. "
            f"The attached images are ordered at {times}. Focus on the middle interval. "
            "Use only visible evidence. Choose one activity from lying_in_bed, sitting_on_bed, "
            "sitting_outside_bed, standing, walking, unknown; and bed_occupancy from "
            "in_bed, out_of_bed, unknown. Return unknown for both when obscured or ambiguous. "
            "Do not infer a fall, emergency, or intent. Reply with JSON only: "
            "activity, bed_occupancy, confidence (0 to 1), reason (one short sentence)."
        )
