# Prerequisites

## Installed project environment

```text
Python 3.12
  ├── MediaPipe + OpenCV      pose and video processing
  ├── Ollama Python SDK       local VLM access
  ├── NumPy + Pandas          data and annotations
  ├── scikit-learn            evaluation metrics
  ├── Matplotlib + Seaborn    report charts
  └── Ruff + Coverage         quality gates
```

Create or restore the locked environment:

```powershell
uv sync --python 3.12
```

Verify it:

```powershell
uv run --frozen ruff check .
uv run --frozen coverage run -m unittest discover -s tests
uv run --frozen coverage report
```

## Ollama

Required model:

```text
qwen3-vl:4b-instruct
```

Default API endpoint:

```text
http://localhost:11434
```

The Python application uses the HTTP endpoint and does not require the Ollama
executable to be on `PATH`.

## Dataset

Fresh-clone download cache:

```text
data/cache/gmdcsa24/GMDCSA24-v2.0.zip
data/cache/gmdcsa24/source/
```

Curated project subset:

```text
data\raw\gmdcsa24
```

The curated subset contains 28 readable MP4 files (15 development, 13
evaluation), totaling 262.7 seconds. Each copied file was verified against its
source with SHA-256.

Official release:

```text
DOI:      10.5281/zenodo.12921216
Size:     1,107,543,412 bytes
MD5:      49bf4eb15a84cc84cb0a4f9c6ddd59e6
Dataset record license:  CC-BY-4.0
Repository file license: MIT
```

Both cache and raw videos are local-only and must not be committed. A fresh
project needs no manually configured dataset path: the command downloads the
official archive only when the complete prepared subset is unavailable under
`data/raw/gmdcsa24`.

Run `uv run elderly-care-agent prepare-dataset` to verify or recreate the
curated subset. See [`DATA_PIPELINE.md`](DATA_PIPELINE.md).
