# Elderly Care Agent

Hybrid computer vision + local Ollama VLM for elderly activity monitoring.

## Sprint 0 foundation + dataset pipeline — complete

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
| Language | Python 3.12 |
| VLM | `qwen3-vl:4b-instruct` through Ollama |
| Tests | standard-library `unittest` |
| Design | OOP, immutable domain models, ports/adapters, SOLID/KISS |

### Setup and test

```powershell
uv sync
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run coverage run -m unittest discover -s tests
uv run coverage report
```

### Demonstrate Sprint 0

```powershell
uv run elderly-care-agent show-config
uv run elderly-care-agent validate-annotations examples/ground_truth.example.json
```

### Prepare and verify GMDCSA-24

```powershell
uv run elderly-care-agent prepare-dataset
```

On a fresh clone this downloads into ignored `data/cache/`, verifies the source,
and creates `data/raw/gmdcsa24`. Later runs return `already_ready` without
downloading or extracting when the prepared subset is complete.

Progress is shown in the terminal and saved to
`logs/elderly-care-agent.log`. Use detailed per-file logging when needed:

```powershell
uv run elderly-care-agent --log-level DEBUG prepare-dataset
```

```text
official archive/local source
            ↓
checksum + layout → tracked selection → atomic staging → OpenCV validation
                                                        ↓
                                              report + SHA-256 inventory
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
See [prerequisites and local dataset layout](docs/PREREQUISITES.md).
See [the dataset pipeline](docs/DATA_PIPELINE.md) and
[dataset ownership/license card](docs/DATASET_CARD.md).

### Ollama prerequisite for Sprint 4

The model is recorded in configuration now; it is not downloaded or called in
Sprint 0.

```powershell
ollama pull qwen3-vl:4b-instruct
```

Large datasets and generated artifacts belong under ignored `data/` and
`outputs/` folders and must not be committed.
