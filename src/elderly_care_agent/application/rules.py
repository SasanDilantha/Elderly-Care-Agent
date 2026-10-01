"""Conservative rules for posture and movement evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median

from elderly_care_agent.config import RuleSettings
from elderly_care_agent.domain.enums import ActivityState, BedOccupancy
from elderly_care_agent.domain.video import FrameReference
from elderly_care_agent.domain.vision import BedRelation, PersonStatus, PoseLandmark, VisionFrame


@dataclass(frozen=True, slots=True)
class RuleObservation:
    reference: FrameReference
    activity: ActivityState
    bed_occupancy: BedOccupancy
    confidence: float
    reason: str
    leg_ratio: float | None
    anchor_speed_per_sec: float | None


class RuleBasedStateClassifier:
    """Classifies only geometry with enough visible evidence."""

    _LEG_JOINTS = ((23, 25, 27), (24, 26, 28))

    def __init__(self, settings: RuleSettings | None = None) -> None:
        self._settings = settings or RuleSettings()

    def classify_all(
        self, frames: tuple[VisionFrame, ...], width: int, height: int
    ) -> tuple[RuleObservation, ...]:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        observations: list[RuleObservation] = []
        previous: VisionFrame | None = None
        for frame in frames:
            observations.append(self.classify(frame, previous, width, height))
            previous = frame
        return tuple(observations)

    def classify(
        self, frame: VisionFrame, previous: VisionFrame | None, width: int, height: int
    ) -> RuleObservation:
        ratio = self._leg_ratio(frame.landmarks, width, height)
        speed = self._anchor_speed(frame, previous)

        def result(
            activity: ActivityState,
            occupancy: BedOccupancy,
            confidence: float,
            reason: str,
        ) -> RuleObservation:
            return RuleObservation(
                frame.reference, activity, occupancy, confidence, reason, ratio, speed
            )

        unknown = ActivityState.UNKNOWN, BedOccupancy.UNKNOWN, 0.0
        if frame.person_status is not PersonStatus.DETECTED:
            return result(*unknown, "pose_not_detected")
        if frame.body_anchor is None or frame.torso_angle_deg is None:
            return result(*unknown, "torso_not_visible")

        if frame.torso_angle_deg <= self._settings.lying_max_torso_angle_deg:
            if frame.bed_relation is BedRelation.INSIDE:
                return result(ActivityState.LYING_IN_BED, BedOccupancy.IN_BED, 0.7, "lying")
            return result(*unknown, "lying_without_bed_evidence")

        if frame.torso_angle_deg < self._settings.upright_min_torso_angle_deg:
            return result(*unknown, "transition_posture")

        if (
            speed is not None
            and speed >= self._settings.walking_min_anchor_speed_per_sec
            and frame.bed_relation is BedRelation.OUTSIDE
            and previous is not None
            and previous.bed_relation is BedRelation.OUTSIDE
            and ratio is not None
        ):
            return result(ActivityState.WALKING, BedOccupancy.OUT_OF_BED, 0.6, "moving_upright")

        if ratio is None:
            return result(*unknown, "legs_not_visible")
        if ratio <= self._settings.sitting_max_leg_ratio:
            if frame.bed_relation is BedRelation.INSIDE:
                return result(ActivityState.SITTING_ON_BED, BedOccupancy.IN_BED, 0.7, "sitting")
            if frame.bed_relation is BedRelation.OUTSIDE:
                return result(
                    ActivityState.SITTING_OUTSIDE_BED, BedOccupancy.OUT_OF_BED, 0.7, "sitting"
                )
            return result(*unknown, "bed_region_unknown")

        if ratio >= self._settings.standing_min_leg_ratio:
            occupancy = (
                BedOccupancy.OUT_OF_BED
                if frame.bed_relation is BedRelation.OUTSIDE
                else BedOccupancy.UNKNOWN
            )
            return result(ActivityState.STANDING, occupancy, 0.7, "standing")
        return result(*unknown, "leg_geometry_ambiguous")

    def _leg_ratio(
        self, landmarks: tuple[PoseLandmark, ...], width: int, height: int
    ) -> float | None:
        visible = {
            point.index: point
            for point in landmarks
            if point.visibility >= self._settings.minimum_landmark_confidence
            and point.presence >= self._settings.minimum_landmark_confidence
            and 0 <= point.x <= 1
            and 0 <= point.y <= 1
        }
        ratios: list[float] = []
        for hip_index, knee_index, ankle_index in self._LEG_JOINTS:
            if not all(index in visible for index in (hip_index, knee_index, ankle_index)):
                continue
            hip, knee, ankle = (visible[index] for index in (hip_index, knee_index, ankle_index))
            thigh = hypot((hip.x - knee.x) * width, (hip.y - knee.y) * height)
            shin = hypot((knee.x - ankle.x) * width, (knee.y - ankle.y) * height)
            if shin >= 1.0:
                ratios.append(thigh / shin)
        return median(ratios) if ratios else None

    @staticmethod
    def _anchor_speed(frame: VisionFrame, previous: VisionFrame | None) -> float | None:
        if previous is None or frame.body_anchor is None or previous.body_anchor is None:
            return None
        elapsed = frame.reference.timestamp_sec - previous.reference.timestamp_sec
        if elapsed <= 0:
            return None
        return (
            hypot(
                frame.body_anchor.x - previous.body_anchor.x,
                frame.body_anchor.y - previous.body_anchor.y,
            )
            / elapsed
        )
