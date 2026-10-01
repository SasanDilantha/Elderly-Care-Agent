"""Ports for a replaceable pose backend and its model asset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from elderly_care_agent.domain.vision import PoseLandmark


class PoseEstimator[FramePayload](ABC):
    @abstractmethod
    def estimate(self, frame: FramePayload, timestamp_sec: float) -> tuple[PoseLandmark, ...]:
        """Return at most one pose for the timestamped frame."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    def __enter__(self) -> PoseEstimator[FramePayload]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PoseEstimatorFactory[FramePayload](ABC):
    @abstractmethod
    def create(
        self,
        model_path: Path,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> PoseEstimator[FramePayload]:
        """Create one estimator for a video analysis run."""


class PoseModelProvider(ABC):
    @abstractmethod
    def ensure_available(self) -> Path:
        """Return a verified local task bundle, downloading when absent."""
