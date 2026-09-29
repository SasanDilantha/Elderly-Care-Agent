# Testable implementation plan

```text
Sprint 0  Foundation          JSON label -> validation -> duration report
Sprint 1  Video input         video -> metadata -> timestamped frames
Sprint 2  Vision features     frame -> person/pose/bed features
Sprint 3  Rules + timeline    features -> stable state segments
Sprint 4  Ollama VLM agent    ambiguous segment -> context frames -> VLM result
Sprint 5  Events + decisions  state transition -> bed event -> NORMAL/MONITOR/ALERT
Sprint 6  Evaluation          prediction + ground truth -> metrics + charts
Sprint 7  Final delivery      video -> report, timeline, annotated video
```

| Sprint | Deliverable | Automated acceptance test |
|---|---|---|
| **0 — complete** | Labels, immutable domain objects, JSON annotation contract, fixed VLM settings, CLI | `unittest` validates invariants, schema, reports, and CLI |
| 1 | `VideoSource` and timestamped frame sampler | Synthetic video returns correct metadata, frame count, and timestamps |
| 2 | Replaceable person/pose/bed feature extractors | Fixture frames produce typed features; missing/occluded person returns `UNKNOWN` safely |
| 3 | Rule classifier, smoothing, duration accumulator | Golden feature sequences produce expected states and exact durations |
| 4 | Ollama adapter for `qwen3-vl:4b-instruct` and context-frame reasoner | Mock HTTP contract test plus opt-in local Ollama smoke test |
| 5 | Bed exit/return state machine and decision policy | Transition tables cover normal, monitor, alert, bounce, and cooldown cases |
| 6 | Accuracy, per-class metrics, confusion matrix, duration error, event timing | Known prediction/ground-truth fixtures produce hand-calculated metrics |
| 7 | CLI pipeline and generated report/annotated video | One short local video passes the end-to-end smoke test |

## Architecture boundaries

```text
CLI / future UI
      |
Application services  <---- ports (interfaces)
      |                          ^
Domain models                    |
                                 |
JSON / OpenCV / pose / Ollama adapters
```

- **Domain** has no framework or I/O dependency.
- **Application** coordinates use cases through ports.
- **Infrastructure** implements file, video, vision, and Ollama adapters.
- **CLI** remains a thin composition and delivery layer.
- Dependencies are added only in the sprint that uses them.

## Definition of done for every sprint

```text
[ ] Feature implemented behind a clear interface
[ ] Happy path + edge cases covered by unittest
[ ] All earlier tests still pass
[ ] One user-visible command or artifact can demonstrate the result
[ ] README and sprint plan updated
[ ] No downloaded videos, secrets, or generated outputs committed
```

