# Requirement checklist

Reference: ASE_AIML_Assignment.pdf. This checklist distinguishes implemented behavior
from measured performance. Functional coverage does not imply high recognition accuracy.

| Requirement | Implementation / evidence |
|---|---|
| Core states | `State`, `ActivityRules`; OUT_OF_BED is the overlapping bed-status field |
| Temporal understanding | `Timeline` rejects brief candidates; `ContextAgent` examines both neighbors |
| Bed exit | `BedEvents`: in-bed history, outside walking, increasing distance, sustained evidence |
| Return to bed | Outside history, renewed occupancy, confirmed lying |
| Sitting up is not exit | Temporal regression tests and the sit-up evaluation clip |
| Activity duration | Sum of contiguous segments, including UNKNOWN; partition test |
| Timeline | `report.json` and `timeline.txt` from transitions |
| Agentic context | Action log shows previous/following observations and selective VLM requests |
| NORMAL / MONITOR / ALERT | `AlertPolicy`; thresholds and reasons in README and settings |
| Event output | Start, confirmation, previous/current state, confidence, decision |
| Observation summary | Durations, occupancy, exits/returns, longest outside interval, final state |
| Activity evaluation | Exact interval-overlap accuracy and confusion, including UNKNOWN |
| Event evaluation | One-to-one matching; TP, FP, FN, precision, recall |
| Duration evaluation | Actual/predicted seconds, signed and absolute error for each state |
| Three failure examples | Measured cases and time intervals in `docs/EVALUATION.md` |
| Source + README + architecture | Active `src/`, root README with Mermaid diagram |
| Simple execution | One CLI; no frontend or service deployment required |
| Familiar libraries | YOLO/Ultralytics, OpenCV, cvzone PoseDetector and drawing utilities |
| OOP and modularity | Small classes with direct method calls; no abstract service hierarchy |
| Minimal comments | Descriptive methods; coordinate-system comment where necessary |
| Reproducibility | Python 3.12, `uv.lock`, fixed evaluation list, committed numeric reports |

## Difficult-case coverage

- Turning, sit-up and brief standing: synthetic temporal regression sequences; real
  sit-up and stand-near-bed clips in the evaluation pilot.
- Exit, return, walking and sitting outside: annotated pilot and curated development clips.
- Missing joints/blankets: missing-visibility regression input; no dedicated blanket video
  has been independently annotated, so blanket accuracy is unmeasured.
- Occlusion and lighting: `scripts/robustness.py` generates reproducible development-video
  variants and records predictions. These are smoke cases, not a labelled benchmark.
- Caregiver: target-lock tests plus a duplicated-person video proxy. The proxy does not
  establish performance for real caregiver interactions or crossings.
- Leaving the camera: UNKNOWN and event-reset regression sequences. Disappearance alone
  cannot establish an exit or prolonged absence.

## Submission contents

`submission/` contains measured reports, timelines, evaluation metrics and the VLM check.
Raw videos and generated robustness clips stay under ignored paths. Prior implementation
files are preserved in `archive/`; the active package and tests are the submission entrypoint.

No email or external publication is performed. The assignment's email submission instruction
is a final action for the candidate after reviewing the code and results.
