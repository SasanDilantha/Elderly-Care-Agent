import unittest

from elderly_care_agent.evaluation import Evaluator
from elderly_care_agent.models import Segment, State
from elderly_care_agent.pipeline import CareMonitor


class EvaluationTests(unittest.TestCase):
    def test_accuracy_counts_unknown_and_duration_errors(self):
        truth = dict(
            duration_sec=4,
            events=[],
            segments=[
                dict(start_sec=0, end_sec=2, activity="lying_in_bed", bed_occupancy="in_bed"),
                dict(start_sec=2, end_sec=4, activity="walking", bed_occupancy="out_of_bed"),
            ],
        )
        report = CareMonitor.summarize(
            [Segment(0, 2, State.LYING, 0.8), Segment(2, 4, State.UNKNOWN, 0)], [], 4
        )
        score = Evaluator.score(truth, report)
        self.assertEqual(score["activity_accuracy"], 0.5)
        self.assertEqual(score["duration_errors"]["walking"]["error"], -2)
        self.assertEqual(score["confusion_seconds"]["walking"]["unknown"], 2)

    def test_duplicate_events_are_false_positives(self):
        truth = dict(
            duration_sec=4,
            segments=[dict(start_sec=0, end_sec=4, activity="walking", bed_occupancy="out_of_bed")],
            events=[dict(event_type="bed_exit", start_sec=1)],
        )
        events = [dict(event="bed_exit", start_sec=1), dict(event="bed_exit", start_sec=1.2)]
        report = CareMonitor.summarize([Segment(0, 4, State.WALKING, 0.8)], events, 4)
        self.assertEqual(
            Evaluator.score(truth, report)["events"]["bed_exit"], dict(tp=1, fp=1, fn=0)
        )

    def test_invalid_annotation_is_rejected(self):
        truth = dict(
            duration_sec=4,
            events=[],
            segments=[dict(start_sec=1, end_sec=4, activity="walking", bed_occupancy="out_of_bed")],
        )
        report = CareMonitor.summarize([Segment(0, 4, State.WALKING, 0.8)], [], 4)
        with self.assertRaises(ValueError):
            Evaluator.score(truth, report)
