# Elderly Care Agent

Hybrid computer vision + local Ollama VLM for elderly activity monitoring.

## Current capabilities

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

The current system validates annotations, prepares the GMDCSA-24 dataset,
samples timestamped video frames, and extracts pose and bed-region geometry.
Activity decisions and Ollama inference are not connected yet.

### Fixed choices

| Area | Choice |
|---|---|
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Language | Python 3.12 |
| VLM | `qwen3-vl:4b-instruct` through Ollama |
| Automated checks | standard-library `unittest` |
| Design | OOP, immutable domain models, ports/adapters, SOLID/KISS |

### Setup

```powershell
uv sync
```

### Validate annotations

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

### Inspect and sample a video

```powershell
uv run elderly-care-agent inspect-video `
  "data\raw\gmdcsa24\development\Subject 1\01.mp4" `
  --sample-fps 2
```

```text
video → OpenCV metadata → deterministic frame indices → decoded frames
                                                   ↓
                                      frame index + media timestamp
```

The command reports metadata and sampled frame references without saving
duplicate frame images.

### Extract vision features

```powershell
uv run elderly-care-agent prepare-vision-model
uv run elderly-care-agent analyze-video `
  "data\raw\gmdcsa24\development\Subject 1\01.mp4" `
  --sample-fps 2 `
  --bed-region 0.10,0.47,0.86,0.87
```

`--bed-region` is a camera-specific rectangle in normalized image coordinates:
`left,top,right,bottom`, where the top-left corner is `0,0` and bottom-right is
`1,1`. The command downloads the pinned MediaPipe Lite model into ignored
`data/cache/vision/` when needed. It reports pose landmarks, a torso anchor,
torso angle, and whether that anchor falls within the configured bed rectangle.
Without a reliable pose or bed rectangle, the bed relation is `unknown`. A
geometric `inside` result is not yet a bed-occupancy decision.

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
  application/     use cases, video and vision services, and interfaces
  infrastructure/  JSON, dataset, logging, OpenCV, and MediaPipe adapters
  cli.py            thin command-line delivery layer
tests/              unittest suite
examples/           small version-controlled annotation fixture
docs/               technical architecture, prerequisites, and data provenance
```

See [prerequisites and local dataset layout](docs/PREREQUISITES.md).
See [the dataset pipeline](docs/DATA_PIPELINE.md) and
[dataset ownership/license card](docs/DATASET_CARD.md).

### Ollama model prerequisite

The model is recorded in configuration but is not called by the current
video-input command.

```powershell
ollama pull qwen3-vl:4b-instruct
```

Large datasets and generated artifacts belong under ignored `data/` and
`outputs/` folders and must not be committed.
