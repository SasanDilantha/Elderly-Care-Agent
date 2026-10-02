import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from elderly_care_agent.activity import ActivityRules
from elderly_care_agent.models import State
from elderly_care_agent.vision import Vision


class VisionTests(unittest.TestCase):
    def vision(self):
        vision = Vision.__new__(Vision)
        vision.target_id = None
        vision.bed_history = []
        vision.bed_box = None
        vision.bed_polygon = None
        vision.next_bed_scan = 0.0
        vision.model = Mock(names={0: "person"})
        vision.model.track.return_value = [SimpleNamespace(boxes=[])]
        vision.bed_model = Mock(return_value=[SimpleNamespace(boxes=[])])
        return vision

    def test_no_pose_from_blanket_darkness_or_occlusion_is_unknown(self):
        rules = ActivityRules([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(rules.classify(0, None, 1).state, State.UNKNOWN)
        self.assertEqual(rules.classify(1, np.zeros((33, 3)), 1).state, State.UNKNOWN)

    def test_angles_are_independent_of_image_aspect_ratio(self):
        a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        self.assertAlmostEqual(ActivityRules.angle(a, b, c), 90)

    def test_caregiver_does_not_replace_target(self):
        vision = self.vision()
        vision.target_id = 7
        box = SimpleNamespace(cls=[0], xyxy=np.array([[0, 0, 10, 20]]), id=[8])
        vision.model = Mock(names={0: "person"})
        vision.model.track.return_value = [SimpleNamespace(boxes=[box])]
        self.assertIsNone(vision.detect(np.zeros((30, 30, 3), dtype=np.uint8)))
        self.assertEqual(vision.target_id, 7)

    def test_ambiguous_people_abstain_without_requesting_selection(self):
        vision = self.vision()
        boxes = [SimpleNamespace(cls=[0], xyxy=np.array([[0, 0, 10, 20]]), id=[i]) for i in [7, 8]]
        vision.model = Mock(names={0: "person"})
        vision.model.track.return_value = [SimpleNamespace(boxes=boxes)]
        self.assertIsNone(vision.detect(np.zeros((30, 30, 3), dtype=np.uint8)))
        self.assertIsNone(vision.target_id)

    def test_patient_on_bed_is_selected_automatically(self):
        vision = self.vision()
        vision.bed_box = (0, 10, 100, 100)
        vision.select_person([((10, 15, 50, 90), 7), ((110, 0, 140, 100), 8)])
        self.assertEqual(vision.target_id, 7)

    def test_equal_bed_overlap_does_not_guess_patient_identity(self):
        vision = self.vision()
        vision.bed_box = (0, 10, 100, 100)
        vision.select_person([((10, 15, 50, 90), 7), ((50, 15, 90, 90), 8)])
        self.assertIsNone(vision.target_id)

    def test_missing_bed_is_retried_and_detected_without_coordinates(self):
        vision = self.vision()
        box = SimpleNamespace(xyxy=np.array([[10, 20, 90, 80]]))
        vision.bed_model.side_effect = [
            [SimpleNamespace(boxes=[])],
            [SimpleNamespace(boxes=[box])],
        ]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vision.detect(frame, 0)
        self.assertIsNone(vision.bed_polygon)
        vision.detect(frame, 1)
        np.testing.assert_array_equal(vision.bed_polygon, [[10, 20], [90, 20], [90, 50], [10, 50]])
        self.assertEqual(vision.bed_model.call_count, 2)
        self.assertEqual(vision.model.track.call_args.kwargs["classes"], [0])

    def test_bed_box_uses_consensus_and_stops_after_five_detections(self):
        vision = self.vision()
        vision.bed_model.side_effect = [
            [SimpleNamespace(boxes=[SimpleNamespace(xyxy=np.array([[x, 20, 90, 80]]))])]
            for x in [10, 12, 11, 10, 9]
        ]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for time in range(6):
            vision.detect(frame, time)
        self.assertEqual(vision.bed_box, (10, 20, 90, 80))
        self.assertEqual(vision.bed_model.call_count, 5)

    def test_multiple_beds_remain_unknown(self):
        vision = self.vision()
        box = SimpleNamespace(xyxy=np.array([[10, 20, 90, 80]]))
        vision.bed_model.return_value = [SimpleNamespace(boxes=[box, box])]
        vision.locate_bed(np.zeros((100, 100, 3)))
        self.assertIsNone(vision.bed_polygon)
        self.assertEqual(ActivityRules().classify(0, np.ones((33, 3)), 1).state, State.UNKNOWN)
