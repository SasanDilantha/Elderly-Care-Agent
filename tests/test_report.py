import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from elderly_care_agent.cli import main
from elderly_care_agent.events import BedEvents
from elderly_care_agent.models import Segment, State
from elderly_care_agent.pipeline import CareMonitor
from elderly_care_agent.report import Report


class ReportTests(unittest.TestCase):
    def setUp(self):
        event = BedEvents.event(
            "bed_exit", 308, 320.75, State.BED_SITTING, State.WALKING, 0.92, "MONITOR"
        )
        self.result = CareMonitor.summarize([Segment(0, 1200, State.UNKNOWN, 0)], [event], 1200)
        self.result["observations"] = [dict(time=0, reason="occluded torso")]

    def test_event_matches_assignment_schema(self):
        self.assertEqual(
            Report(self.result).events,
            [
                dict(
                    event="bed_exit",
                    start_time="00:05:08",
                    confirm_time="00:05:20",
                    previous_state="sitting_on_bed",
                    current_state="walking",
                    confidence=0.92,
                    decision="MONITOR",
                )
            ],
        )

    def test_summary_has_only_assignment_fields(self):
        summary = Report(self.result).summary
        self.assertEqual(
            list(summary),
            [
                "observation_duration_sec",
                "activity_duration_sec",
                "bed_exit_count",
                "bed_return_count",
                "total_in_bed_sec",
                "total_out_of_bed_sec",
                "longest_out_of_bed_period_sec",
                "final_state",
            ],
        )
        self.assertEqual(sum(summary["activity_duration_sec"].values()), 1200)
        self.assertEqual(summary["total_out_of_bed_sec"], 0)

    def test_additional_file_preserves_diagnostics_and_precise_event_times(self):
        with tempfile.TemporaryDirectory() as directory:
            Report(self.result).save(directory)
            files = {
                name: json.loads((Path(directory) / name).read_text())
                for name in ("report.json", "events.json", "additional.json")
            }
        self.assertEqual({**files["report.json"], **files["additional.json"]}, self.result)
        self.assertEqual(files["additional.json"]["events"][0]["confirmed_sec"], 320.75)
        self.assertEqual(files["events.json"][0]["confirm_time"], "00:05:20")

    def test_timestamps_preserve_elapsed_hours(self):
        self.assertEqual(Report.timestamp(59.99), "00:00:59")
        self.assertEqual(Report.timestamp(60), "00:01:00")
        self.assertEqual(Report.timestamp(3600), "01:00:00")
        self.assertEqual(Report.timestamp(90000), "25:00:00")

    def test_cli_prints_expected_output_before_additional_file_links(self):
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("sys.argv", ["elderly-care-agent", "analyze", "clip.mp4", "--output", directory]),
            patch.object(CareMonitor, "run", return_value=self.result),
            redirect_stdout(output),
        ):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(list(result), ["summary", "events", "additional_outputs"])
        self.assertNotIn("observations", result["summary"])
        self.assertEqual(result["events"][0]["start_time"], "00:05:08")
