import csv
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


class Dataset:
    def prepare(self, source=None):
        with Path("data/manifests/gmdcsa24_selection.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        root = Path("data/raw/gmdcsa24")
        targets = [root / row["split"] / row["subject"] / row["video"] for row in rows]
        if all(path.is_file() and path.stat().st_size > 0 for path in targets):
            return dict(status="already_ready", videos=len(targets))
        if source is None:
            import yaml

            config = yaml.safe_load(Path("config/dataset.yml").read_text(encoding="utf-8"))[
                "dataset"
            ]
            cache = Path("data/cache/gmdcsa24")
            cache.mkdir(parents=True, exist_ok=True)
            archive = cache / "GMDCSA24-v2.0.zip"
            if not archive.exists():
                temporary = archive.with_suffix(".download")
                urllib.request.urlretrieve(config["source"]["archive_url"], temporary)
                temporary.replace(archive)
            with archive.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "md5").hexdigest()
            if checksum != config["source"]["archive_checksum"].split(":", 1)[1]:
                raise ValueError("Dataset archive checksum mismatch")
            source = cache / "source"
            source.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    destination = (source / entry.filename).resolve()
                    if not destination.is_relative_to(source.resolve()):
                        raise ValueError("Unsafe path in dataset archive")
                bundle.extractall(source)
        for row, target in zip(rows, targets, strict=True):
            if target.is_file() and target.stat().st_size > 0:
                continue
            matches = [
                path
                for path in Path(source).rglob(row["video"])
                if path.parent.name == row["category"] and path.parent.parent.name == row["subject"]
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one source video for {row['subject']}/{row['video']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], target)
        return dict(status="prepared", videos=len(targets))
