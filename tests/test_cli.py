import io
import json
from pathlib import Path
import unittest

from elderly_care_agent.cli import CliApplication


class CliApplicationTests(unittest.TestCase):
    def test_show_config_prints_required_model(self) -> None:
        output = io.StringIO()

        exit_code = CliApplication(output=output).run(["show-config"])

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("qwen3-vl:4b-instruct", payload["vlm"]["model"])

    def test_validate_annotations_prints_summary(self) -> None:
        output = io.StringIO()

        exit_code = CliApplication(output=output).run(
            ["validate-annotations", str(Path("examples/ground_truth.example.json"))]
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertTrue(payload["valid"])
        self.assertEqual(4, payload["segment_count"])
        self.assertEqual(10.0, payload["occupancy_durations_sec"]["in_bed"])


if __name__ == "__main__":
    unittest.main()

