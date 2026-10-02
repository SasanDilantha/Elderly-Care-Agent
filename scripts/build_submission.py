import json
from pathlib import Path

from elderly_care_agent.models import State


def main():
    source = Path("outputs/automatic_evaluation")
    target = Path("submission")
    target.mkdir(exist_ok=True)
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for case in metrics["cases"]:
        destination = target / case["id"]
        destination.mkdir(exist_ok=True)
        for name in ["report.json", "events.json", "additional.json", "timeline.txt"]:
            (destination / name).write_bytes((source / case["id"] / name).read_bytes())
    duration = {state: dict(actual=0.0, predicted=0.0, absolute_error=0.0) for state in State}
    confusion = {state: {} for state in State}
    for case in metrics["cases"]:
        for state, values in case["duration_errors"].items():
            for key in duration[state]:
                duration[state][key] += values[key]
        for actual, predicted in case["confusion_seconds"].items():
            for label, seconds in predicted.items():
                confusion[actual][label] = confusion[actual].get(label, 0) + seconds
    lines = [
        "# Evaluation: measured assignment prototype",
        "",
        "Run on 2026-10-02 with the locked project environment, YOLO26s and cvzone/MediaPipe.",
        "Person tracking uses every decoded frame; pose/activity use 5 samples/second.",
        "Bed/contact-region and person selection are automatic; no supplied coordinates or IDs.",
        f"The five clips total {metrics['duration']:.2f} seconds. "
        "These results are vision + temporal context only.",
        "",
        "| Metric | Measured result |",
        "|---|---:|",
        "| Activity accuracy, weighted by time | %.2f%% |" % (100 * metrics["activity_accuracy"]),
        "| Bed-status accuracy, weighted by time | %.2f%% |" % (100 * metrics["bed_accuracy"]),
    ]
    for name, counts in metrics["events"].items():
        precision = (
            "undefined (no predictions)"
            if counts["precision"] is None
            else "%.2f%%" % (100 * counts["precision"])
        )
        recall = "undefined" if counts["recall"] is None else "%.2f%%" % (100 * counts["recall"])
        lines.append(f"| {name}: TP / FP / FN | {counts['tp']} / {counts['fp']} / {counts['fn']} |")
        lines.append(f"| {name}: precision / recall | {precision} / {recall} |")
    lines += ["", "## Per-clip results", "", "| Clip | Activity accuracy |", "|---|---:|"]
    lines += [
        f"| {case['id']} | {100 * case['activity_accuracy']:.2f}% |" for case in metrics["cases"]
    ]
    lines += [
        "",
        "## Duration comparison",
        "",
        "Absolute errors below are summed across clips, so opposite errors do not cancel.",
        "",
        "| Activity | Annotated seconds | Predicted seconds | Sum absolute error |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {state} | {row['actual']:.2f} | {row['predicted']:.2f} | {row['absolute_error']:.2f} |"
        for state, row in duration.items()
    ]
    lines += [
        "",
        "## Confusion and failure cases",
        "",
        "The complete time-weighted confusion matrices and per-clip errors are in",
        "[metrics.json](../submission/metrics.json). The following are observed disagreements",
        "with the provisional annotation; they are not hypothetical examples.",
        "",
    ]
    explanations = {
        "subject3_03_bed_exit": "Pose changes during sitting up leave a gap in activity "
        "recognition, even though the later bed exit is detected.",
        "subject3_12_stand_near_bed": "Hip movement from bending or projection resembles "
        "displacement; the rule confuses standing with walking.",
        "subject4_08_sit_up_no_exit": "The sit-up transition has ambiguous joint geometry. "
        "No exit is predicted.",
        "subject4_10_bed_return": "The estimated bed region and occluded pose leave "
        "insufficient lying evidence; "
        "the expected return is missed.",
    }
    for case in metrics["cases"]:
        if not case["failures"]:
            continue
        failure = max(case["failures"], key=lambda row: row["end"] - row["start"])
        lines.append(
            f"- **{case['id']}, {failure['start']:.2f}-{failure['end']:.2f}s:** "
            f"annotated `{failure['actual']}`, predicted `{failure['predicted']}`. "
            + explanations[case["id"]]
        )
    lines += [
        "",
        "## Method and limits",
        "",
        "Accuracy is the sum of correctly labelled overlap seconds divided by annotation duration.",
        "UNKNOWN is included. Events match one-to-one by type and start time within one second.",
        "No predictions means undefined precision, never 100%. "
        "Missed events are explicitly counted.",
        "The exit detector uses repeated visible-foot movement and sustained walking; hip overlap",
        "with the bed is insufficient to rule out an exit in a projected camera image.",
        "Brief posture gaps retain history only while the same person remains tracked.",
        "The positive return remains missed because its outside and lying evidence is unreliable.",
        "",
        "Labels were previously estimated from frames sampled about every 0.5 seconds. They are",
        "provisional, not independently adjudicated. Development uses Subjects 1-2; these clips",
        "use Subjects 3-4. Evaluation clips were inspected while fixing tracking bugs, so these",
        "numbers are a development pilot, not an unbiased final generalization benchmark.",
        "",
        "No clip establishes performance on 60-second absence alerts, blankets, or real caregiver",
        "interactions. Those policy/identity behaviors have regression tests; generated stress",
        "videos are smoke cases, not clinical or independently labelled evidence.",
        "",
        "Reproduce: `uv run elderly-care-agent evaluate --output outputs/automatic_evaluation`",
        "then `uv run python scripts/build_submission.py`.",
        "",
    ]
    Path("docs/EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "confusion_seconds.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )
    for origin, name in [
        ("outputs/vlm_validation.json", "vlm_validation.json"),
        ("outputs/robustness/results.json", "robustness.json"),
    ]:
        if Path(origin).exists():
            (target / name).write_bytes(Path(origin).read_bytes())
    print("Saved numerical reports and evaluation documentation to submission/ and docs/.")


if __name__ == "__main__":
    main()
