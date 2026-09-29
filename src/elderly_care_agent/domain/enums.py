"""Stable labels shared by annotation, inference, and evaluation layers."""

from enum import Enum


class ActivityState(str, Enum):
    LYING_IN_BED = "lying_in_bed"
    SITTING_ON_BED = "sitting_on_bed"
    SITTING_OUTSIDE_BED = "sitting_outside_bed"
    STANDING = "standing"
    WALKING = "walking"
    UNKNOWN = "unknown"


class BedOccupancy(str, Enum):
    IN_BED = "in_bed"
    OUT_OF_BED = "out_of_bed"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    BED_EXIT = "bed_exit"
    BED_RETURN = "bed_return"


class Decision(str, Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    ALERT = "alert"


class ObservationSource(str, Enum):
    RULES = "rules"
    VLM = "vlm"
    FUSED = "fused"
    GROUND_TRUTH = "ground_truth"

