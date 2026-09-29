# Elderly Care Agent

Hybrid computer vision + local Ollama VLM for elderly activity monitoring.

## Sprint 0 — complete

```text
Ground-truth JSON
       |
       v
JSON repository -> validated domain timeline -> duration/report service -> CLI JSON
                         |
                         +-- activity state
                         +-- bed occupancy
                         +-- bed exit/return event
```

This sprint intentionally contains **no video inference**. It establishes a small,
testable base before OpenCV, pose estimation, and Ollama are introduced.

### Fixed choices

| Area | Choice |
|---|---|
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Language | Python 3.12–3.14 |
| VLM | `qwen3-vl:4b-instruct` through Ollama |
| Tests | standard-library `unittest` |
| Design | OOP, immutable domain models, ports/adapters, SOLID/KISS |

### Setup and test

```powershell
uv sync
uv run python -m unittest discover -s tests -v
```

### Demonstrate Sprint 0

```powershell
uv run elderly-care-agent show-config
uv run elderly-care-agent validate-annotations examples/ground_truth.example.json
```

Expected validation result: `valid: true`, 4 segments, 1 bed-exit event,
10 seconds in bed, and 10 seconds out of bed.

### Project structure

```text
src/elderly_care_agent/
  domain/          labels and validated immutable models
  application/     use cases and dependency interfaces
  infrastructure/  JSON adapter; video/vision/Ollama adapters come later
  cli.py            thin command-line delivery layer
tests/              unittest suite
examples/           small version-controlled annotation fixture
docs/               testable sprint plan
```

See [the complete sprint plan](docs/SPRINT_PLAN.md).

### Ollama prerequisite for Sprint 4

The model is recorded in configuration now; it is not downloaded or called in
Sprint 0.

```powershell
ollama pull qwen3-vl:4b-instruct
```

Large datasets and generated artifacts belong under ignored `data/` and
`outputs/` folders and must not be committed.
