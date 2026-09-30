# Local datasets

```text
data/
├── manifests/       tracked selection and provenance files
└── raw/             ignored video files; never commit
```

On a fresh clone, the complete archive and extracted source are downloaded to
ignored project cache directories:

```text
data/cache/gmdcsa24/
├── GMDCSA24-v2.0.zip
└── source/
```

The curated subset used by this project is copied to:

```text
data\raw\gmdcsa24
```

The tracked selection is defined in
`data/manifests/gmdcsa24_selection.csv`:

```text
development = Subjects 1 and 2
evaluation  = Subjects 3 and 4
```

Keeping people separated between the two splits prevents subject leakage.

```text
Curated videos:   28
Development:      15 videos
Evaluation:       13 videos
Total duration:   262.7 seconds
Copied size:      216.7 MB
```

Source: GMDCSA-24 v2.0, DOI `10.5281/zenodo.12921216`.

Prepare or verify this layout with the idempotent pipeline:

```powershell
uv run elderly-care-agent prepare-dataset
```

```text
data/raw/gmdcsa24 complete? ── yes ──> skip download and preparation
              │
              no
              ↓
download → checksum → extract cache → select/split → data/raw/gmdcsa24
```

The cache and raw folders are ignored by Git. No machine-specific absolute path
is required. Set `GMDCSA24_CACHE_DIR` only when an optional external cache is
preferred.

The pipeline configuration is tracked in `config/dataset.yml`; its report and
per-video SHA-256 inventory are generated under ignored `artifacts/data_pipeline/`.
See [`docs/DATASET_CARD.md`](../docs/DATASET_CARD.md) for ownership and licensing.

Do not modify source videos. Any normalized or generated files belong under
`data/processed/` or `outputs/`, which are also ignored by Git.
