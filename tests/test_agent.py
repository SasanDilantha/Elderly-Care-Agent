import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from elderly_care_agent.agent import ContextAgent
from elderly_care_agent.models import Observation, Segment, Settings, State
from elderly_care_agent.pipeline import CareMonitor


class AgentTests(unittest.TestCase):
    def test_context_asks_for_both_sides(self):
        agent = ContextAgent(Settings(), use_vlm=True)
        agent.ask = Mock(
            return_value=dict(state=State.BED_SITTING, confidence=0.9, reason="seated")
        )
        segments = [
            Segment(0, 1, State.LYING, 0.8),
            Segment(1, 2, State.UNKNOWN, 0),
            Segment(2, 3, State.BED_SITTING, 0.8),
        ]
        result = agent.review("clip.mp4", segments, [Observation(1, person_id=4)])
        self.assertEqual(result[1].source, "vlm")
        self.assertEqual(agent.ask.call_args.args[2].state, State.LYING)
        self.assertEqual(agent.ask.call_args.args[3].state, State.BED_SITTING)

    def test_unverified_boundary_proposal_does_not_fill_gap(self):
        agent = ContextAgent(Settings(), use_vlm=True)
        agent.ask = Mock(return_value=dict(state=State.WALKING, confidence=0.99))
        segments = [
            Segment(0, 1, State.LYING, 0.8),
            Segment(1, 2, State.UNKNOWN, 0),
            Segment(2, 3, State.WALKING, 0.8),
        ]
        self.assertEqual(
            agent.review("", segments, [Observation(1, person_id=4)])[1].state, State.UNKNOWN
        )

    def test_vlm_is_bounded_and_invalid_video_abstains(self):
        agent = ContextAgent(Settings(max_reviews=1), use_vlm=True)
        agent.ask = Mock(return_value=dict(state=State.UNKNOWN, confidence=0))
        segments = [
            Segment(0, 1, State.UNKNOWN, 0),
            Segment(1, 2, State.WALKING, 0.8),
            Segment(2, 3, State.UNKNOWN, 0),
        ]
        agent.review("", segments, [])
        self.assertEqual(agent.ask.call_count, 1)
        result = ContextAgent(Settings()).ask("missing.mp4", segments[0], None, None, 3)
        self.assertEqual(result["state"], State.UNKNOWN)

    def test_report_files_and_duration_are_valid_json(self):
        report = CareMonitor.summarize([Segment(0, 2, State.UNKNOWN, 0)], [], 2)
        with tempfile.TemporaryDirectory() as directory:
            CareMonitor.save(report, directory)
            self.assertIn("UNKNOWN", (Path(directory) / "timeline.txt").read_text())
            self.assertTrue((Path(directory) / "report.json").exists())

    def test_video_failure_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with (
            patch("elderly_care_agent.pipeline.cv2.VideoCapture", return_value=capture),
            self.assertRaises(ValueError),
        ):
            CareMonitor().run("missing.mp4", [0, 0, 1, 1])
        capture.release.assert_called_once()
