from dataclasses import asdict, dataclass
from enum import StrEnum


class State(StrEnum):
    LYING = "lying_in_bed"
    BED_SITTING = "sitting_on_bed"
    SITTING = "sitting_outside_bed"
    STANDING = "standing"
    WALKING = "walking"
    UNKNOWN = "unknown"

    @property
    def bed(self):
        if self in {State.LYING, State.BED_SITTING}:
            return "in_bed"
        return "unknown" if self == State.UNKNOWN else "out_of_bed"


@dataclass
class Observation:
    time: float
    state: State = State.UNKNOWN
    confidence: float = 0.0
    distance: float | None = None
    person_id: int | None = None
    reason: str = "no reliable pose"
    features: dict | None = None


@dataclass
class Segment:
    start: float
    end: float
    state: State
    confidence: float
    source: str = "vision"

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        return {**asdict(self), "bed_state": self.state.bed}


@dataclass
class Settings:
    sample_fps: float = 5.0
    confirmation: float = 0.4
    bridge_gap: float = 0.6
    event_hold: float = 0.4
    unknown_monitor: float = 3.0
    sitting_monitor: float = 120.0
    absence_alert: float = 60.0
    vlm_model: str = "qwen3-vl:4b-instruct"
    vlm_timeout: float = 120.0
    max_reviews: int = 3
