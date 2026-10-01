"""Validated VLM evidence for an uncertain video interval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from elderly_care_agent.domain.enums import ActivityState, BedOccupancy, ObservationSource
from elderly_care_agent.domain.exceptions import DomainValidationError
from elderly_care_agent.domain.models import TimeRange
from elderly_care_agent.domain.video import FrameReference


class ReviewStatus(StrEnum):
    PROPOSED = "proposed"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class VlmReview:
    time_range: TimeRange
    status: ReviewStatus
    activity: ActivityState
    bed_occupancy: BedOccupancy
    self_reported_confidence: float
    reason: str
    context_frames: tuple[FrameReference, ...]
    source: ObservationSource = ObservationSource.VLM

    def __post_init__(self) -> None:
        if not self.context_frames:
            raise DomainValidationError("VLM review requires context frames")
        if (
            not isfinite(self.self_reported_confidence)
            or not 0 <= self.self_reported_confidence <= 1
        ):
            raise DomainValidationError("VLM confidence must be within [0, 1]")
        if not self.reason.strip() or len(self.reason) > 500:
            raise DomainValidationError("VLM reason must contain 1 to 500 characters")
        if self.status is ReviewStatus.ABSTAINED:
            if (
                self.activity is not ActivityState.UNKNOWN
                or self.bed_occupancy is not BedOccupancy.UNKNOWN
                or self.self_reported_confidence != 0
            ):
                raise DomainValidationError("abstained VLM review must be unknown")
        elif self.activity is ActivityState.UNKNOWN or self.bed_occupancy is BedOccupancy.UNKNOWN:
            raise DomainValidationError("proposed VLM review must name activity and occupancy")
        elif (
            self.activity in {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}
            and self.bed_occupancy is not BedOccupancy.IN_BED
        ) or (
            self.activity
            in {
                ActivityState.SITTING_OUTSIDE_BED,
                ActivityState.STANDING,
                ActivityState.WALKING,
            }
            and self.bed_occupancy is not BedOccupancy.OUT_OF_BED
        ):
            raise DomainValidationError("VLM activity and occupancy are inconsistent")
