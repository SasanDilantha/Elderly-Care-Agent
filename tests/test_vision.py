import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from elderly_care_agent.activity import ActivityRules
from elderly_care_agent.models import State
from elderly_care_agent.vision import Vision


class VisionTests(unittest.TestCase):
    def test_no_pose_from_blanket_darkness_or_occlusion_is_unknown(self):
        rules = ActivityRules([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(rules.classify(0, None, 1).state, State.UNKNOWN)
        self.assertEqual(rules.classify(1, np.zeros((33, 3)), 1).state, State.UNKNOWN)

    def test_angles_are_independent_of_image_aspect_ratio(self):
        a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        self.assertAlmostEqual(ActivityRules.angle(a, b, c), 90)

    def test_caregiver_does_not_replace_target(self):
        vision = Vision.__new__(Vision)
        vision.target_id = 7
        vision.bed_boxes = []
        box = SimpleNamespace(cls=[0], xyxy=np.array([[0, 0, 10, 20]]), id=[8])
        vision.model = Mock(names={0: "person"})
        vision.model.track.return_value = [SimpleNamespace(boxes=[box])]
        self.assertIsNone(vision.detect(np.zeros((30, 30, 3), dtype=np.uint8)))
        self.assertEqual(vision.target_id, 7)

    def test_multiple_people_require_target_selection(self):
        vision = Vision.__new__(Vision)
        vision.target_id = None
        vision.bed_boxes = []
        boxes = [SimpleNamespace(cls=[0], xyxy=np.array([[0, 0, 10, 20]]), id=[i]) for i in [7, 8]]
        vision.model = Mock(names={0: "person"})
        vision.model.track.return_value = [SimpleNamespace(boxes=boxes)]
        self.assertIsNone(vision.detect(np.zeros((30, 30, 3), dtype=np.uint8)))
        self.assertIsNone(vision.target_id)

    def test_invalid_bed_region_is_rejected(self):
        with self.assertRaises(ValueError):
            Vision.bed_region(np.zeros((30, 30, 3)), [0.8, 0, 0.2, 1])
