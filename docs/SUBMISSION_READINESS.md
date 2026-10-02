# Submission readiness

Reviewed on 2026-10-02 against all 13 pages of `ASE_AIML_Assignment.pdf`.

The local submission contains all nine requested deliverable types. It is a working
prototype with disclosed recognition failures, not a fully validated recognition system.
The assignment does not specify a minimum accuracy score. The positive return-to-bed
video is still missed, and several difficult cases have only synthetic coverage.

## Deliverables

| Requested deliverable | Where to find it | Status |
|---|---|---|
| Source code | [src/elderly_care_agent](../src/elderly_care_agent/) | Modular Python classes; YOLO, OpenCV and cvzone |
| README | [README](../README.md) | Setup, use, assumptions and limitations |
| Simple architecture diagram | [Architecture](../README.md#architecture) | Mermaid diagram in README |
| Run instructions | [Install and run](../README.md#install) | Locked Python 3.12 environment and CLI |
| Activity timeline | [Example timeline](../submission/subject3_03_bed_exit/timeline.txt) | Five measured timelines supplied |
| Activity-duration summary | [Example summary](../submission/subject3_03_bed_exit/report.json) | Five summaries; activity totals equal observation duration |
| Bed-exit/return events | [Real exit](../submission/subject3_03_bed_exit/events.json), [synthetic return/alert](../submission/scenario_events.json) | Real exit detected; real positive return missed and reported |
| Evaluation results | [Evaluation](EVALUATION.md), [metrics](../submission/metrics.json) | Accuracy, confusion, duration errors, event precision/recall |
| At least three failure examples | [Four timestamped failures](EVALUATION.md#confusion-and-failure-cases) | Observed disagreements with provisional annotations |

The primary JSON follows section 7 of the PDF. The event key is `confirmed_time`.
`report.json` has the eight requested summary fields. Diagnostic features, unknown
bed time and precise numeric event timestamps are separate in `additional.json`.
`out_of_bed` is an overlapping bed status rather than a seventh exclusive activity;
this prevents counting walking time twice. Confidence values are heuristic.

## Final checks

- 40 unit tests pass; Ruff lint and formatting pass.
- `uv sync --frozen --offline` succeeds in the existing environment. A fresh machine
  needs internet to download dependencies, YOLO weights and the dataset.
- CLI help and dataset preparation succeed; all 28 selected local videos are present.
- Source distribution and wheel build successfully. They exclude raw videos, weights
  and the local virtual environment.
- All five submitted reports are valid JSON. Event schemas, event counts, contiguous
  timelines and activity/bed duration partitions were checked.
- The latest full exit-video run confirms departure at 9.821 seconds. The real
  stand-near-bed and sit-up clips produce no false exit.
- Optional Ollama integration has one recorded successful local request. It has not
  been evaluated as an accuracy improvement across the five clips.

Reproduce the software checks from the repository root:

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run elderly-care-agent prepare-dataset
uv run elderly-care-agent analyze "data/raw/gmdcsa24/evaluation/Subject 3/03.mp4" --show
```

Reproduce the measured reports:

```powershell
uv run elderly-care-agent evaluate --output outputs/automatic_evaluation
uv run python scripts/build_submission.py
uv run python scripts/event_demo.py
```

## Limits to disclose

- The five-video pilot totals only 56.46 seconds. Activity accuracy is 54.51% and
  bed-status accuracy is 60.52%. There is one positive exit and one positive return:
  the exit is detected, the return is missed. Exit precision/recall of 100% on this
  sample does not establish general reliability.
- Labels are approximate. Evaluation videos were inspected during debugging, so the
  results are a development pilot, not an untouched final test set.
- Turning and brief stand-then-sit behavior have synthetic temporal tests. Blankets,
  actual caregiver interactions, leaving/re-entering view and prolonged absence lack
  independently annotated real-video validation. Chair sitting and room walking are
  selected in the dataset but are not separately scored in the five-clip pilot.
- Low light, blackout and duplicated-person clips exercise failure handling; their
  results are smoke checks rather than labelled accuracy measurements.
- Bed contact uses an approximate region from a detected rectangle. Pose occlusion,
  camera perspective and tracking loss can still prevent correct states and events.

These are remaining recognition/validation gaps. They must not be described as fully
solved merely because the files and software checks are complete.

## Publication and submission

The reviewed branch is `feat/clean-assignment-submission`. At the remote check on
2026-10-02, this branch was not present on GitHub. The root repository link therefore
does not yet give a reviewer this final version.

When ready to publish the reviewed local branch:

```powershell
git push -u origin feat/clean-assignment-submission
```

Use its branch URL in the email, or merge it into the default branch before using the
root repository URL. Confirm the chosen link opens for the recipient; repository
visibility and recipient access have not been verified.

The assignment requests an email to `careers@newnop.com` with subject
`ASE AI/ML Assignment - Sasan Dilantha`. The following draft is ready to review after
publishing. It has not been sent.

> Hello,
>
> Please find my ASE AI/ML assignment at:
> https://github.com/SasanDilantha/Elderly-Care-Agent/tree/feat/clean-assignment-submission
>
> The README includes setup instructions and the architecture diagram. The submission
> directory contains timelines, summaries, events and evaluation artifacts, with four
> failure cases explained in docs/EVALUATION.md.
>
> This prototype uses YOLO, OpenCV, cvzone pose estimation and temporal rules, with
> optional local VLM review. The small pilot achieves 54.51% activity accuracy; it
> detects the positive bed exit but misses the positive return. With more time I would
> improve bed-contact estimation and tracking, then evaluate on independently labelled
> videos with caregivers, occlusion and blankets.
>
> Regards,
> Sasan Dilantha

Before the interview, practise explaining and changing the video loop, pose rules,
temporal confirmation and event logic without coding assistance. See
[implementation explanation](EXPLANATION.md).
