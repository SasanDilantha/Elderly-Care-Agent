import hashlib
import shutil
import tempfile
import unittest
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from elderly_care_agent.application.dataset_ports import (
    DatasetDownloader,
    DatasetExtractor,
    VideoProbe,
)
from elderly_care_agent.components.data_ingestion import DataIngestion
from elderly_care_agent.domain.dataset import (
    DatasetIngestionConfig,
    DatasetSource,
    VideoProbeResult,
)
from elderly_care_agent.domain.exceptions import DatasetIntegrityError
from elderly_care_agent.infrastructure.csv_dataset_selection_repository import (
    CsvDatasetSelectionRepository,
)
from elderly_care_agent.infrastructure.dataset_io import SafeZipDatasetExtractor


class UnexpectedDownloader(DatasetDownloader):
    def download(self, source_url: str, destination: Path) -> None:
        raise AssertionError(f"download was not expected: {source_url} -> {destination}")


class UnexpectedExtractor(DatasetExtractor):
    def extract(self, archive_path: Path, destination: Path) -> None:
        raise AssertionError(f"extraction was not expected: {archive_path} -> {destination}")


class CopyingDownloader(DatasetDownloader):
    def __init__(self, source_archive: Path) -> None:
        self.source_archive = source_archive
        self.calls = 0

    def download(self, source_url: str, destination: Path) -> None:
        self.calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source_archive, destination)


class FakeVideoProbe(VideoProbe):
    def inspect(self, video_path: Path) -> VideoProbeResult:
        if not video_path.read_bytes().startswith(b"video-"):
            raise DatasetIntegrityError("fake decoder rejected content")
        return VideoProbeResult(frame_count=90, fps=30.0, duration_sec=3.0)


class DataIngestionTests(unittest.TestCase):
    def test_prepares_existing_source_idempotently_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = self._create_source(root / "source")
            config = self._config(root, source_dir, expected_video_count=2, expected_csv_count=2)
            ingestion = self._ingestion(config)

            first_report = ingestion.run()
            shutil.rmtree(source_dir)
            second_report = ingestion.run()

            self.assertEqual("existing_source_directory", first_report.source_mode)
            self.assertEqual(2, first_report.copied_video_count)
            self.assertEqual(0, first_report.reused_video_count)
            self.assertEqual("already_ready", second_report.status)
            self.assertEqual("prepared_output", second_report.source_mode)
            self.assertEqual(0, second_report.copied_video_count)
            self.assertEqual(2, second_report.reused_video_count)
            self.assertEqual(
                b"video-development",
                (config.output_dir / "development/Subject 1/01.mp4").read_bytes(),
            )
            self.assertTrue(config.report_path.is_file())
            inventory = config.inventory_path.read_text(encoding="utf-8")
            self.assertIn("sha256", inventory)
            self.assertIn("evaluation/Subject 3/02.mp4", inventory)

    def test_force_revalidates_and_replaces_a_changed_staged_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = self._create_source(root / "source")
            config = self._config(root, source_dir, expected_video_count=2, expected_csv_count=2)
            ingestion = self._ingestion(config)
            ingestion.run()
            staged_video = config.output_dir / "development/Subject 1/01.mp4"
            staged_video.write_bytes(b"changed")

            skipped_report = ingestion.run()
            self.assertEqual("already_ready", skipped_report.status)
            self.assertEqual(b"changed", staged_video.read_bytes())
            report = ingestion.run(force=True)

            self.assertEqual(b"video-development", staged_video.read_bytes())
            self.assertEqual(1, report.copied_video_count)
            self.assertEqual(1, report.reused_video_count)

    def test_downloads_and_extracts_when_raw_dataset_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture_archive = root / "fixture/data.zip"
            fixture_archive.parent.mkdir()
            with zipfile.ZipFile(fixture_archive, "w") as zip_file:
                zip_file.writestr("release/LICENSE", "MIT")
                zip_file.writestr("release/README.md", "source")
                zip_file.writestr("release/Subject 1/ADL.csv", "File Name,Description\n")
                zip_file.writestr("release/Subject 1/ADL/01.mp4", b"video-development")
            checksum = hashlib.md5(fixture_archive.read_bytes()).hexdigest()  # noqa: S324
            manifest = root / "selection.csv"
            manifest.write_text(
                "split,subject,category,video,coverage,event_focus\n"
                "development,Subject 1,ADL,01.mp4,walking,none\n",
                encoding="utf-8",
            )
            config = self._config(
                root,
                root / "resource/source",
                expected_video_count=1,
                expected_csv_count=1,
                subjects=("Subject 1",),
                manifest=manifest,
                archive=root / "data/cache/data.zip",
                checksum=f"md5:{checksum}",
            )
            config.archive_path.parent.mkdir(parents=True, exist_ok=True)
            config.archive_path.write_bytes(b"truncated-cache")
            downloader = CopyingDownloader(fixture_archive)
            ingestion = DataIngestion(
                config,
                downloader,
                SafeZipDatasetExtractor(),
                CsvDatasetSelectionRepository(),
                FakeVideoProbe(),
                clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
            )

            report = ingestion.run()

            self.assertEqual("downloaded_archive", report.source_mode)
            self.assertEqual(1, downloader.calls)
            self.assertEqual(fixture_archive.read_bytes(), config.archive_path.read_bytes())
            self.assertTrue((config.source_dir / "Subject 1/ADL/01.mp4").is_file())

    @staticmethod
    def _create_source(source_dir: Path) -> Path:
        (source_dir / "Subject 1/ADL").mkdir(parents=True)
        (source_dir / "Subject 3/ADL").mkdir(parents=True)
        (source_dir / "LICENSE").write_text("MIT", encoding="utf-8")
        (source_dir / "README.md").write_text("source", encoding="utf-8")
        (source_dir / "Subject 1/ADL.csv").write_text("File Name,Description\n")
        (source_dir / "Subject 3/ADL.csv").write_text("File Name,Description\n")
        (source_dir / "Subject 1/ADL/01.mp4").write_bytes(b"video-development")
        (source_dir / "Subject 3/ADL/02.mp4").write_bytes(b"video-evaluation")
        return source_dir

    @staticmethod
    def _source(checksum: str) -> DatasetSource:
        return DatasetSource(
            owner="Ekram Alam",
            canonical_repository="https://example.test/upstream",
            mirror_repository="https://example.test/mirror",
            mirror_commit="a" * 40,
            archive_url="https://example.test/data.zip",
            archive_checksum=checksum,
            dataset_doi="10.5281/zenodo.12921216",
            article_doi="10.1016/j.dib.2024.110892",
            dataset_license="CC-BY-4.0",
            repository_license="MIT",
        )

    @classmethod
    def _config(
        cls,
        root: Path,
        source_dir: Path,
        expected_video_count: int,
        expected_csv_count: int,
        subjects: tuple[str, ...] = ("Subject 1", "Subject 3"),
        manifest: Path | None = None,
        archive: Path | None = None,
        checksum: str = f"md5:{'0' * 32}",
    ) -> DatasetIngestionConfig:
        if manifest is None:
            manifest = root / "selection.csv"
            manifest.write_text(
                "split,subject,category,video,coverage,event_focus\n"
                "development,Subject 1,ADL,01.mp4,walking,bed_exit\n"
                "evaluation,Subject 3,ADL,02.mp4,lying_in_bed,bed_return\n",
                encoding="utf-8",
            )
        return DatasetIngestionConfig(
            dataset_name="GMDCSA-24",
            dataset_version="v2.0",
            source=cls._source(checksum),
            archive_path=archive or root / "resource/data.zip",
            source_dir=source_dir,
            selection_manifest=manifest,
            output_dir=root / "raw/gmdcsa24",
            report_path=root / "artifacts/report.json",
            inventory_path=root / "artifacts/inventory.csv",
            expected_video_count=expected_video_count,
            expected_metadata_csv_count=expected_csv_count,
            expected_subjects=subjects,
        )

    @staticmethod
    def _ingestion(config: DatasetIngestionConfig) -> DataIngestion:
        return DataIngestion(
            config,
            UnexpectedDownloader(),
            UnexpectedExtractor(),
            CsvDatasetSelectionRepository(),
            FakeVideoProbe(),
            clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
        )


if __name__ == "__main__":
    unittest.main()
