import unittest

from elderly_care_agent.agent import ContextAgent
from elderly_care_agent.events import AlertPolicy, BedEvents
from elderly_care_agent.models import Observation, Segment, Settings, State
from elderly_care_agent.pipeline import CareMonitor
from elderly_care_agent.timeline import Timeline


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def segments(self, *states):
        return [Segment(i * 2, (i + 1) * 2, state, 0.8) for i, state in enumerate(states)]

    def observation(self, time, state, foot_distance, person_id=1):
        return Observation(
            time,
            state,
            distance=0.2,
            person_id=person_id,
            features=dict(foot_distance=foot_distance, torso_length=1.0),
        )

    def test_durations_partition_observation(self):
        segments = self.segments(State.LYING, State.UNKNOWN, State.WALKING, State.SITTING)
        report = CareMonitor.summarize(segments, [], 8)
        self.assertAlmostEqual(sum(report["activity_duration_sec"].values()), 8)
        self.assertAlmostEqual(
            report["total_in_bed_sec"]
            + report["total_out_of_bed_sec"]
            + report["total_bed_unknown_sec"],
            8,
        )
        self.assertEqual(report["longest_out_of_bed_period_sec"], 4)

    def test_timeline_removes_flicker_without_losing_time(self):
        rows = [
            Observation(0, State.LYING),
            Observation(1, State.STANDING),
            Observation(1.2, State.LYING),
            Observation(2, State.LYING),
        ]
        segments = Timeline(self.settings).build(rows, 3)
        self.assertEqual([s.state for s in segments], [State.LYING, State.UNKNOWN, State.LYING])
        self.assertEqual(sum(s.duration for s in segments), 3)

    def test_short_gap_requires_same_identity(self):
        segments = [
            Segment(0, 1, State.LYING, 0.8),
            Segment(1, 1.4, State.UNKNOWN, 0),
            Segment(1.4, 3, State.LYING, 0.8),
        ]
        agent = ContextAgent(self.settings)
        result = agent.review("", segments, [Observation(1, person_id=None)])
        self.assertEqual(result[1].state, State.UNKNOWN)
        result = agent.review("", segments, [Observation(1, person_id=1)])
        self.assertEqual(result[1].state, State.LYING)
        self.assertEqual(result[1].source, "context")

    def test_turning_sitting_up_and_brief_stand_are_not_exits(self):
        for states in [
            (State.LYING, State.LYING),
            (State.LYING, State.BED_SITTING),
            (State.BED_SITTING, State.STANDING, State.BED_SITTING),
        ]:
            with self.subTest(states=states):
                self.assertEqual(BedEvents(self.settings).detect(self.segments(*states), []), [])

    def test_departure_needs_moving_away(self):
        segments = self.segments(State.BED_SITTING, State.STANDING, State.WALKING)
        rows = [
            self.observation(2, State.STANDING, -0.5),
            self.observation(4, State.WALKING, -1),
            self.observation(4.2, State.WALKING, -1.1),
            self.observation(4.4, State.WALKING, -1.2),
        ]
        events = BedEvents(self.settings).detect(segments, rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "bed_exit")
        for row in rows:
            row.features["foot_distance"] = -0.5
        self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])

    def test_brief_visible_uncertainty_preserves_exit_history(self):
        segments = [
            Segment(0, 2, State.BED_SITTING, 0.8),
            Segment(2, 3, State.WALKING, 0.8),
            Segment(3, 3.603, State.UNKNOWN, 0),
            Segment(3.603, 5, State.WALKING, 0.8),
        ]
        rows = [
            self.observation(0, State.BED_SITTING, -0.5),
            self.observation(2, State.WALKING, -0.5),
            self.observation(3, State.UNKNOWN, None),
            self.observation(3.603, State.WALKING, -1),
            self.observation(3.803, State.WALKING, -1.1),
            self.observation(4.003, State.WALKING, -1.2),
        ]
        events = BedEvents(self.settings).detect(segments, rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_sec"], 2)
        self.assertAlmostEqual(events[0]["confirmed_sec"], 4.003)
        for identity in [None, 2]:
            rows[2].person_id = identity
            self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])

    def test_long_visible_gap_resets_exit_history(self):
        segments = self.segments(State.BED_SITTING, State.UNKNOWN, State.WALKING)
        rows = [
            self.observation(0, State.BED_SITTING, -0.5),
            self.observation(2, State.UNKNOWN, None),
            self.observation(4, State.WALKING, -0.5),
            self.observation(4.5, State.WALKING, -1),
            self.observation(5, State.WALKING, -1.2),
        ]
        self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])

    def test_exit_confirms_when_movement_threshold_is_reached_late(self):
        segments = self.segments(State.BED_SITTING, State.WALKING)
        rows = [
            self.observation(2, State.WALKING, -0.5),
            self.observation(2.2, State.WALKING, -0.7),
            self.observation(2.4, State.WALKING, -0.8),
            self.observation(2.6, State.WALKING, -0.85),
        ]
        events = BedEvents(self.settings).detect(segments, rows)
        self.assertEqual(events[0]["confirmed_sec"], 2.6)

    def test_moving_feet_briefly_or_without_walking_is_not_exit(self):
        segments = self.segments(State.BED_SITTING, State.WALKING)
        rows = [
            self.observation(2, State.WALKING, -0.5),
            self.observation(2.2, State.WALKING, -1),
            self.observation(2.4, State.WALKING, -0.5),
        ]
        self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])
        for row in rows:
            row.state = State.STANDING
        rows[-1].time, rows[-1].features["foot_distance"] = 3, -1.5
        self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])

    def test_return_requires_lying_after_sitting(self):
        segments = self.segments(State.WALKING, State.BED_SITTING)
        self.assertEqual(BedEvents(self.settings).detect(segments, []), [])
        segments.append(Segment(4, 6, State.LYING, 0.8))
        events = BedEvents(self.settings).detect(segments, [])
        self.assertEqual(events[0]["event"], "bed_return")
        self.assertEqual(events[0]["start_sec"], 2)
        self.assertEqual(events[0]["confirmed_sec"], 4.4)

    def test_inferred_walking_without_observed_steps_cannot_confirm_exit(self):
        segments = self.segments(State.BED_SITTING, State.STANDING, State.WALKING)
        segments[-1].source = "context"
        rows = [
            self.observation(2, State.STANDING, -0.5),
            self.observation(3, State.STANDING, -1),
            self.observation(4, State.UNKNOWN, -1),
        ]
        self.assertEqual(BedEvents(self.settings).detect(segments, rows), [])

    def test_leaving_camera_is_not_exit(self):
        segments = self.segments(State.LYING, State.UNKNOWN, State.WALKING)
        self.assertEqual(BedEvents(self.settings).detect(segments, []), [])

    def test_vlm_cannot_create_event(self):
        segments = self.segments(State.LYING, State.WALKING)
        segments[1].source = "vlm"
        self.assertEqual(BedEvents(self.settings).detect(segments, []), [])

    def test_prolonged_unknown_is_monitor_not_alert(self):
        decision = AlertPolicy(self.settings).decide([Segment(0, 90, State.UNKNOWN, 0)], [])
        self.assertEqual(decision["decision"], "MONITOR")

    def test_prolonged_sitting_is_monitor(self):
        result = AlertPolicy(self.settings).decide([Segment(0, 121, State.BED_SITTING, 0.8)], [])
        self.assertEqual(result["decision"], "MONITOR")

    def test_alert_requires_observed_exit_and_continuous_absence(self):
        segments = [Segment(0, 70, State.WALKING, 0.8)]
        event = dict(event="bed_exit", confirmed_sec=1)
        policy = AlertPolicy(self.settings)
        self.assertEqual(policy.decide(segments, [])["decision"], "NORMAL")
        self.assertEqual(policy.decide(segments, [event])["decision"], "ALERT")
        segments = [
            Segment(0, 30, State.WALKING, 0.8),
            Segment(30, 35, State.UNKNOWN, 0),
            Segment(35, 80, State.WALKING, 0.8),
        ]
        self.assertEqual(policy.decide(segments, [event])["decision"], "MONITOR")


if __name__ == "__main__":
    unittest.main()
