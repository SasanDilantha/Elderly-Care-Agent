import tempfile
import unittest
from pathlib import Path

from elderly_care_agent.domain.exceptions import DatasetIntegrityError
from elderly_care_agent.infrastructure.csv_dataset_selection_repository import (
    CsvDatasetSelectionRepository,
)


class CsvDatasetSelectionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = CsvDatasetSelectionRepository()

    def test_loads_valid_subject_isolated_selection(self) -> None:
        manifest = self._manifest(
            "split,subject,category,video,coverage,event_focus\n"
            "development,Subject 1,ADL,01.mp4,walking|standing,bed_exit\n"
            "evaluation,Subject 3,ADL,02.mp4,lying_in_bed,bed_return\n"
        )

        rows = self.repository.load(manifest)

        self.assertEqual(2, len(rows))
        self.assertEqual(Path("Subject 1/ADL/01.mp4"), rows[0].source_relative_path)
        self.assertEqual(("walking", "standing"), rows[0].coverage)

    def test_rejects_subject_leakage(self) -> None:
        manifest = self._manifest(
            "split,subject,category,video,coverage,event_focus\n"
            "development,Subject 1,ADL,01.mp4,walking,none\n"
            "evaluation,Subject 1,ADL,02.mp4,walking,none\n"
        )

        with self.assertRaisesRegex(DatasetIntegrityError, "cannot cross"):
            self.repository.load(manifest)

    def test_rejects_unsafe_video_path(self) -> None:
        manifest = self._manifest(
            "split,subject,category,video,coverage,event_focus\n"
            "development,Subject 1,ADL,../01.mp4,walking,none\n"
        )

        with self.assertRaisesRegex(DatasetIntegrityError, "unsafe video"):
            self.repository.load(manifest)

    def _manifest(self, content: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        manifest = Path(temporary_directory.name) / "selection.csv"
        manifest.write_text(content, encoding="utf-8")
        return manifest


if __name__ == "__main__":
    unittest.main()
