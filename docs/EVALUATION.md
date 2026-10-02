# Evaluation: measured assignment prototype

Run on 2026-10-02 with the locked project environment, YOLO26s and cvzone/MediaPipe.
Person tracking uses every decoded frame; pose/activity use 5 samples/second.
The five clips total 56.46 seconds. These results are vision + temporal context only.

| Metric | Measured result |
|---|---:|
| Activity accuracy, weighted by time | 59.67% |
| Bed-status accuracy, weighted by time | 67.14% |
| bed_exit: TP / FP / FN | 0 / 0 / 1 |
| bed_exit: precision / recall | undefined (no predictions) / 0.00% |
| bed_return: TP / FP / FN | 0 / 0 / 1 |
| bed_return: precision / recall | undefined (no predictions) / 0.00% |

## Per-clip results

| Clip | Activity accuracy |
|---|---:|
| subject3_03_bed_exit | 58.85% |
| subject3_12_stand_near_bed | 70.32% |
| subject4_08_sit_up_no_exit | 57.53% |
| subject4_10_bed_return | 19.70% |
| subject4_11_sitting_on_floor | 100.00% |

## Duration comparison

Absolute errors below are summed across clips, so opposite errors do not cancel.

| Activity | Annotated seconds | Predicted seconds | Sum absolute error |
|---|---:|---:|---:|
| lying_in_bed | 10.75 | 3.02 | 7.85 |
| sitting_on_bed | 27.39 | 20.51 | 8.02 |
| sitting_outside_bed | 10.72 | 10.72 | 0.00 |
| standing | 5.72 | 1.61 | 7.33 |
| walking | 1.87 | 5.39 | 3.52 |
| unknown | 0.00 | 15.21 | 15.21 |

## Confusion and failure cases

The complete time-weighted confusion matrices and per-clip errors are in
[metrics.json](../submission/metrics.json). The following are observed disagreements
with the provisional annotation; they are not hypothetical examples.

- **subject3_03_bed_exit, 1.81-2.82s:** annotated `sitting_on_bed`, predicted `unknown`. Pose changes during sitting up; uncertain transitions also prevent the annotated exit from being confirmed.
- **subject3_12_stand_near_bed, 9.61-10.67s:** annotated `standing`, predicted `walking`. Hip movement from bending or projection resembles displacement; the rule confuses standing with walking.
- **subject4_08_sit_up_no_exit, 1.81-3.62s:** annotated `sitting_on_bed`, predicted `unknown`. The sit-up transition has ambiguous joint geometry. No exit is predicted.
- **subject4_10_bed_return, 9.42-12.80s:** annotated `lying_in_bed`, predicted `unknown`. Lying and occluded pose evidence is insufficient; the expected return is missed.

## Method and limits

Accuracy is the sum of correctly labelled overlap seconds divided by annotation duration.
UNKNOWN is included. Events match one-to-one by type and start time within one second.
No predictions means undefined precision, never 100%. The missed positive exit and return
are explicitly counted. The event state machine passes synthetic tests, but real-video
event recall in this small pilot is zero. This is a key limitation of the submission.

Labels were previously estimated from frames sampled about every 0.5 seconds. They are
provisional, not independently adjudicated. Development uses Subjects 1-2; these clips
use Subjects 3-4. Evaluation clips were inspected while fixing tracking bugs, so these
numbers are a development pilot, not an unbiased final generalization benchmark.

No clip establishes performance on 60-second absence alerts, blankets, or real caregiver
interactions. Those policy/identity behaviors have regression tests; generated stress
videos are smoke cases, not clinical or independently labelled evidence.

Reproduce: `uv run elderly-care-agent evaluate --output outputs/submission_evaluation`
then `uv run python scripts/build_submission.py`.
