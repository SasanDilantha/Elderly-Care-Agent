# Submission artifacts

- `metrics.json`: measured five-clip evaluation, including confusion, duration errors,
  event TP/FP/FN, and timestamped failure intervals.
- Each case folder: assignment summary in `report.json`, assignment-format events in
  `events.json`, diagnostics in `additional.json`, and readable `timeline.txt`.
  `additional.json` preserves numerical event times and automatic `scene_setup` details.
- `confusion_seconds.json`: aggregate confusion, measured in seconds.
- `robustness.json`: generated low-light, occlusion, and duplicated-person smoke runs.
- `vlm_validation.json`: actual local Ollama response or recorded abstention/error.
- `scenario_events.json`: synthetic event/alert demonstration, clearly separated from
  measured video results. It proves the policy path, not recognition accuracy.

The evaluated prototype misses the pilot's positive exit and return. This limitation
is disclosed in `docs/EVALUATION.md`; synthetic examples must not be substituted for
those measured results. Videos and model weights are excluded from the repository.
