"""Typed application settings for the hybrid vision/VLM pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class VisionSettings:
    """Settings owned by the deterministic computer-vision pipeline."""

    sample_fps: float = 2.0
    minimum_detection_confidence: float = 0.50
    minimum_tracking_confidence: float = 0.50

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be greater than zero")
        for name, value in (
            ("minimum_detection_confidence", self.minimum_detection_confidence),
            ("minimum_tracking_confidence", self.minimum_tracking_confidence),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TemporalSettings:
    """Settings for state smoothing and event confirmation."""

    state_confirmation_sec: float = 1.5
    bed_event_confirmation_sec: float = 2.0

    def __post_init__(self) -> None:
        if not isfinite(self.state_confirmation_sec) or self.state_confirmation_sec <= 0:
            raise ValueError("state_confirmation_sec must be finite and greater than zero")
        if not isfinite(self.bed_event_confirmation_sec) or self.bed_event_confirmation_sec <= 0:
            raise ValueError("bed_event_confirmation_sec must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class RuleSettings:
    """Conservative geometry thresholds for a fixed-camera view."""

    minimum_landmark_confidence: float = 0.5
    lying_max_torso_angle_deg: float = 30.0
    upright_min_torso_angle_deg: float = 55.0
    sitting_max_leg_ratio: float = 0.55
    standing_min_leg_ratio: float = 0.70
    walking_min_anchor_speed_per_sec: float = 0.08

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_landmark_confidence <= 1:
            raise ValueError("minimum_landmark_confidence must be between 0 and 1")
        if not 0 <= self.lying_max_torso_angle_deg < self.upright_min_torso_angle_deg <= 90:
            raise ValueError("torso angle thresholds must be ordered within [0, 90]")
        if not 0 < self.sitting_max_leg_ratio < self.standing_min_leg_ratio:
            raise ValueError("leg ratio thresholds must be positive and ordered")
        if (
            not isfinite(self.walking_min_anchor_speed_per_sec)
            or self.walking_min_anchor_speed_per_sec <= 0
        ):
            raise ValueError("walking_min_anchor_speed_per_sec must be greater than zero")


@dataclass(frozen=True, slots=True)
class VlmSettings:
    """Settings for the local Ollama vision-language model."""

    model: str = "qwen3-vl:4b-instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    context_offsets_sec: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not self.context_offsets_sec or 0.0 not in self.context_offsets_sec:
            raise ValueError("context_offsets_sec must include the current frame (0.0)")
        if tuple(sorted(self.context_offsets_sec)) != self.context_offsets_sec:
            raise ValueError("context_offsets_sec must be sorted")


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """Root configuration object passed to application services."""

    vision: VisionSettings = field(default_factory=VisionSettings)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    rules: RuleSettings = field(default_factory=RuleSettings)
    vlm: VlmSettings = field(default_factory=VlmSettings)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe configuration snapshot."""

        return asdict(self)
