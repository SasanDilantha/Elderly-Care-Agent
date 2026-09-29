import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from elderly_care_agent.domain.enums import ActivityState, EventType
from elderly_care_agent.domain.exceptions import AnnotationFormatError
from elderly_care_agent.infrastructure.json_annotation_repository import (
    JsonAnnotationRepository,
)


class JsonAnnotationRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = JsonAnnotationRepository()

    def test_example_annotation_is_valid(self) -> None:
        path = Path("examples/ground_truth.example.json")

        annotation = self.repository.load(path)

        self.assertEqual("bedroom_demo_001", annotation.video_id)
        self.assertEqual(20.0, annotation.duration_sec)
        self.assertEqual(ActivityState.LYING_IN_BED, annotation.segments[0].activity)
        self.assertEqual(EventType.BED_EXIT, annotation.events[0].event_type)

    def test_invalid_label_has_contextual_error(self) -> None:
        payload = {
            "video_id": "bad",
            "duration_sec": 1.0,
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "activity": "flying",
                    "bed_occupancy": "unknown",
                }
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(AnnotationFormatError, "invalid annotation"):
                self.repository.load(path)


if __name__ == "__main__":
    unittest.main()

