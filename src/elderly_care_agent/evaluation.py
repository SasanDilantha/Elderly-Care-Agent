import json
from collections import defaultdict
from pathlib import Path

import yaml

from elderly_care_agent.models import State
from elderly_care_agent.pipeline import CareMonitor


class Evaluator:
    @staticmethod
    def score(truth, report, tolerance=1.0):
        segments = truth["segments"]
        duration = truth["duration_sec"]
        end = 0.0
        for row in segments:
            if abs(row["start_sec"] - end) > 0.001 or row["end_sec"] <= end:
                raise ValueError(
                    "Ground truth must be contiguous, nonoverlapping, and start at zero"
                )
            State(row["activity"])
            end = row["end_sec"]
        if abs(end - duration) > 0.01 or abs(report["observation_duration_sec"] - duration) > 0.1:
            raise ValueError("Ground-truth and decoded durations do not match")
        confusion = {state.value: defaultdict(float) for state in State}
        bed_correct = 0.0
        errors = []
        for actual in segments:
            for predicted in report["timeline"]:
                start = max(actual["start_sec"], predicted["start"])
                stop = min(actual["end_sec"], predicted["end"])
                overlap = max(0, stop - start)
                confusion[actual["activity"]][predicted["state"]] += overlap
                if actual["bed_occupancy"] == predicted["bed_state"]:
                    bed_correct += overlap
                if overlap and actual["activity"] != predicted["state"]:
                    errors.append(
                        dict(
                            start=start,
                            end=stop,
                            actual=actual["activity"],
                            predicted=predicted["state"],
                        )
                    )
        events = {}
        for kind in ["bed_exit", "bed_return"]:
            expected = [e for e in truth["events"] if e["event_type"] == kind]
            predicted = [e for e in report["events"] if e["event"] == kind]
            available = set(range(len(expected)))
            matched = 0
            for event in predicted:
                possible = [
                    i
                    for i in available
                    if abs(expected[i]["start_sec"] - event["start_sec"]) <= tolerance
                ]
                if possible:
                    best = min(
                        possible, key=lambda i: abs(expected[i]["start_sec"] - event["start_sec"])
                    )
                    available.remove(best)
                    matched += 1
            events[kind] = dict(tp=matched, fp=len(predicted) - matched, fn=len(expected) - matched)
        duration_errors = {}
        for state in State:
            actual = sum(s["end_sec"] - s["start_sec"] for s in segments if s["activity"] == state)
            predicted = report["activity_duration_sec"][state]
            duration_errors[state] = dict(
                actual=actual,
                predicted=predicted,
                error=predicted - actual,
                absolute_error=abs(predicted - actual),
            )
        correct = sum(confusion[state].get(state, 0) for state in State)
        return dict(
            duration=duration,
            correct_seconds=correct,
            activity_accuracy=correct / duration,
            bed_accuracy=bed_correct / duration,
            confusion_seconds=confusion,
            duration_errors=duration_errors,
            events=events,
            failures=errors,
        )

    def run(self, config_path, monitor, directory, use_vlm=False):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        directory = Path(directory)
        cases = []
        for case in config["cases"]:
            print(f"Evaluating {case['id']}", flush=True)
            report = monitor.run(case["video"], case["bed_region"], use_vlm=use_vlm)
            CareMonitor.save(report, directory / case["id"])
            truth = json.loads(Path(case["annotation"]).read_text(encoding="utf-8"))
            score = self.score(truth, report, config["event_match_tolerance_sec"])
            cases.append(dict(id=case["id"], **score))
        total = sum(case["duration"] for case in cases)
        events = {}
        for kind in ["bed_exit", "bed_return"]:
            counts = {
                key: sum(case["events"][kind][key] for case in cases) for key in ["tp", "fp", "fn"]
            }
            counts["precision"] = (
                counts["tp"] / (counts["tp"] + counts["fp"])
                if counts["tp"] + counts["fp"]
                else None
            )
            counts["recall"] = (
                counts["tp"] / (counts["tp"] + counts["fn"])
                if counts["tp"] + counts["fn"]
                else None
            )
            events[kind] = counts
        result = dict(
            dataset=config["dataset"],
            with_vlm=use_vlm,
            duration=total,
            activity_accuracy=sum(case["correct_seconds"] for case in cases) / total,
            bed_accuracy=sum(case["bed_accuracy"] * case["duration"] for case in cases) / total,
            events=events,
            cases=cases,
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
