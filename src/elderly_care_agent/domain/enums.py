"""Stable labels shared by annotation, inference, and evaluation layers."""

from enum import StrEnum


class ActivityState(StrEnum):
    LYING_IN_BED = "lying_in_bed"
    SITTING_ON_BED = "sitting_on_bed"
    SITTING_OUTSIDE_BED = "sitting_outside_bed"
    STANDING = "standing"
    WALKING = "walking"
    UNKNOWN = "unknown"


class BedOccupancy(StrEnum):
    IN_BED = "in_bed"
    OUT_OF_BED = "out_of_bed"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    BED_EXIT = "bed_exit"
    BED_RETURN = "bed_return"


class Decision(StrEnum):
    NORMAL = "normal"
    MONITOR = "monitor"
    ALERT = "alert"


class ObservationSource(StrEnum):
    RULES = "rules"
    VLM = "vlm"
    FUSED = "fused"
    GROUND_TRUTH = "ground_truth"
