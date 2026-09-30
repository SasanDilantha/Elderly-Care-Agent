import csv
import unittest
from pathlib import Path


class Gmdcsa24ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest_path = Path("data/manifests/gmdcsa24_selection.csv")
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            self.rows = list(csv.DictReader(manifest_file))

    def test_selection_contains_unique_video_paths(self) -> None:
        identities = {
            (row["split"], row["subject"], row["category"], row["video"]) for row in self.rows
        }

        self.assertEqual(28, len(self.rows))
        self.assertEqual(len(self.rows), len(identities))
        self.assertEqual({"ADL"}, {row["category"] for row in self.rows})

    def test_subjects_do_not_leak_between_splits(self) -> None:
        development_subjects = {
            row["subject"] for row in self.rows if row["split"] == "development"
        }
        evaluation_subjects = {row["subject"] for row in self.rows if row["split"] == "evaluation"}

        self.assertEqual({"Subject 1", "Subject 2"}, development_subjects)
        self.assertEqual({"Subject 3", "Subject 4"}, evaluation_subjects)
        self.assertTrue(development_subjects.isdisjoint(evaluation_subjects))

    def test_required_scenarios_are_covered(self) -> None:
        coverage = {label for row in self.rows for label in row["coverage"].split("|")}
        events = {row["event_focus"] for row in self.rows}

        self.assertTrue(
            {
                "lying_in_bed",
                "sitting_on_bed",
                "sitting_outside_bed",
                "standing",
                "walking",
                "unknown",
                "out_of_bed",
            }.issubset(coverage)
        )
        self.assertTrue({"bed_exit", "bed_return", "hard_negative"}.issubset(events))


if __name__ == "__main__":
    unittest.main()
