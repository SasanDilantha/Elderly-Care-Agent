# Validation performed

Validated on Windows on 2026-10-02 using the locked Python 3.12 environment.

- CLI entrypoint and help run through `uv run --frozen`.
- 25 unittest cases cover temporal transitions, duration totals, evaluation matching,
  identity selection, unavailable pose, VLM restrictions, output writing and cleanup.
- Ruff lint and formatting checks pass for `src`, `tests` and `scripts`.
- Five real videos completed inference: 59.67% time-weighted activity accuracy and
  67.14% bed-status accuracy. Both annotated positive bed events were missed.
  See [evaluation](EVALUATION.md) for the confusion matrix, durations and failures.
- Saved timelines partition each video's analyzed duration; summaries are checked
  against the final event and policy code.

## Local VLM integration

One live Qwen3-VL 4B request through Ollama returned `sitting_on_bed` with a model-reported
confidence of 0.98 for the development clip `Subject 1/05.mp4`. The single image contains
three chronological panels from 3.0, 4.5 and 6.0 seconds. The result is in
[vlm_validation.json](../submission/vlm_validation.json).

This check used a 180-second timeout; the application default is 120 seconds. It checks
the integration, not VLM accuracy or its impact across the evaluation clips. Confidence
is uncalibrated. Failed and timed-out requests abstain and remain recorded in agent actions.

## Generated stress videos

`scripts/robustness.py` transforms the development clip to exercise difficult inputs.
These have no independent activity labels and do not establish recognition accuracy.

| Transformation | UNKNOWN time | Event predictions | Decision |
|---|---:|---:|---|
| Brightness reduced to 10% | 5.64 s | 0 | MONITOR |
| Blackout from 5 to 7 seconds | 8.97 s | 0 | MONITOR |
| Duplicated person as a two-person proxy | 10.55 s | 0 | MONITOR |

The long uncertainty after blackout exposes lost tracking identity. The two-person proxy
correctly abstains without a selected target, but is not a real caregiver interaction test.
Numeric results are in [robustness.json](../submission/robustness.json).

`scripts/event_demo.py` produces a clearly labelled synthetic example of exit, return and
prolonged-absence decisions. It verifies policy behavior separately from the real-video
event failures; it is not evidence of video recognition accuracy.
