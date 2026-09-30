import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from elderly_care_agent.domain.exceptions import DatasetIntegrityError
from elderly_care_agent.infrastructure.dataset_io import (
    FileHasher,
    HttpDatasetDownloader,
    SafeZipDatasetExtractor,
)


class FakeHttpResponse(io.BytesIO):
    def __init__(self, content: bytes, declared_size: int) -> None:
        super().__init__(content)
        self.headers = {"Content-Length": str(declared_size)}


class FileHasherTests(unittest.TestCase):
    def test_verifies_expected_digest_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sample = Path(temporary_directory) / "sample.bin"
            sample.write_bytes(b"known-content")
            digest = FileHasher.digest(sample, "md5")

            FileHasher.verify(sample, f"md5:{digest}")
            with self.assertRaisesRegex(DatasetIntegrityError, "checksum mismatch"):
                FileHasher.verify(sample, f"md5:{'0' * 32}")


class SafeZipDatasetExtractorTests(unittest.TestCase):
    def test_strips_shared_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("release/README.md", "source")
                zip_file.writestr("release/Subject 1/ADL/01.mp4", b"video")

            destination = root / "dataset"
            SafeZipDatasetExtractor().extract(archive, destination)

            self.assertEqual("source", (destination / "README.md").read_text())
            self.assertEqual(b"video", (destination / "Subject 1/ADL/01.mp4").read_bytes())

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../outside.txt", "unsafe")

            with self.assertRaisesRegex(DatasetIntegrityError, "unsafe ZIP member"):
                SafeZipDatasetExtractor().extract(archive, root / "dataset")
            self.assertFalse((root / "outside.txt").exists())


class HttpDatasetDownloaderTests(unittest.TestCase):
    def test_formats_progress_sizes_for_humans(self) -> None:
        self.assertEqual("1.0 KiB", HttpDatasetDownloader.format_bytes(1024))
        self.assertEqual("1.0 GiB", HttpDatasetDownloader.format_bytes(1024**3))

    @patch("elderly_care_agent.infrastructure.dataset_io.urllib.request.urlopen")
    def test_publishes_only_a_complete_download(self, urlopen) -> None:
        urlopen.return_value = FakeHttpResponse(b"complete", declared_size=8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "archive.zip"
            stale_part = root / ".archive.zip.interrupted.part"
            stale_part.write_bytes(b"old")

            HttpDatasetDownloader(max_attempts=1).download(
                "https://example.test/archive.zip",
                destination,
            )

            self.assertEqual(b"complete", destination.read_bytes())
            self.assertFalse(stale_part.exists())

    @patch("elderly_care_agent.infrastructure.dataset_io.urllib.request.urlopen")
    def test_rejects_and_cleans_an_incomplete_download(self, urlopen) -> None:
        urlopen.return_value = FakeHttpResponse(b"short", declared_size=100)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "archive.zip"

            with self.assertRaisesRegex(DatasetIntegrityError, "download failed"):
                HttpDatasetDownloader(max_attempts=1).download(
                    "https://example.test/archive.zip",
                    destination,
                )

            self.assertFalse(destination.exists())
            self.assertEqual([], list(root.glob("*.part")))


if __name__ == "__main__":
    unittest.main()
