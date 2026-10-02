import json
from pathlib import Path

from elderly_care_agent.events import AlertPolicy, BedEvents
from elderly_care_agent.models import Observation, Segment, Settings, State
from elderly_care_agent.pipeline import CareMonitor


def main():
    settings = Settings()
    segments = [
        Segment(0, 5, State.LYING, 0.8),
        Segment(5, 8, State.BED_SITTING, 0.8),
        Segment(8, 10, State.STANDING, 0.8),
        Segment(10, 80, State.WALKING, 0.8),
        Segment(80, 83, State.BED_SITTING, 0.8),
        Segment(83, 90, State.LYING, 0.8),
    ]
    observations = [
        Observation(8, State.STANDING, distance=-0.1),
        Observation(10.5, State.WALKING, distance=-1),
    ]
    events = BedEvents(settings).detect(segments, observations)
    result = CareMonitor.summarize(segments, events, 90)
    result["source"] = "synthetic policy demonstration; not video accuracy evidence"
    result["decision"] = AlertPolicy(settings).decide(segments, events)
    Path("submission").mkdir(exist_ok=True)
    Path("submission/scenario_events.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    assert result["bed_exit_count"] == result["bed_return_count"] == 1
    assert result["decision"]["decision"] == "ALERT"
    print("Synthetic exit, return, duration totals, and prolonged-absence alert verified.")


if __name__ == "__main__":
    main()
