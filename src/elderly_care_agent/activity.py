from collections import deque
from math import acos, atan2, degrees

import cv2
import numpy as np

from elderly_care_agent.models import Observation, State


class ActivityRules:
    def __init__(self, bed=None):
        self.bed = np.asarray(bed, dtype=np.float32) if bed is not None else None
        self.motion = deque()

    @staticmethod
    def angle(a, b, c):
        first, second = a - b, c - b
        length = np.linalg.norm(first) * np.linalg.norm(second)
        return degrees(acos(float(np.clip(first @ second / length, -1, 1)))) if length else None

    def classify(self, time, landmarks, person_id):
        if self.bed is None:
            self.motion.clear()
            return Observation(time, person_id=person_id, reason="bed not detected unambiguously")
        if landmarks is None:
            self.motion.clear()
            return Observation(time, person_id=person_id)
        sides = [(11, 23, 25, 27), (12, 24, 26, 28)]
        side = max(sides, key=lambda ids: min(landmarks[ids[0], 2], landmarks[ids[1], 2]))
        shoulder_id, hip_id, knee_id, ankle_id = side
        if min(landmarks[shoulder_id, 2], landmarks[hip_id, 2]) < 0.5:
            self.motion.clear()
            return Observation(time, person_id=person_id, reason="occluded torso")
        shoulder, hip, knee, ankle = landmarks[list(side), :2]
        scale = float(np.linalg.norm(shoulder - hip))
        if scale < 0.02:
            return Observation(time, person_id=person_id, reason="degenerate torso")
        torso = degrees(atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1])))
        hip_angle = self.angle(shoulder, hip, knee) if landmarks[knee_id, 2] >= 0.5 else None
        knee_angle = (
            self.angle(hip, knee, ankle) if min(landmarks[[knee_id, ankle_id], 2]) >= 0.5 else None
        )
        hips = landmarks[[23, 24]]
        anchor = hips[hips[:, 2] >= 0.5, :2].mean(axis=0)
        self.motion.append((time, anchor.copy(), scale))
        while len(self.motion) > 1 and time - self.motion[0][0] > 0.8:
            self.motion.popleft()
        elapsed = time - self.motion[0][0]
        speed = None
        if elapsed >= 0.3:
            speed = float(np.linalg.norm(anchor - self.motion[0][1]) / scale / elapsed)
        distance = float(cv2.pointPolygonTest(self.bed, tuple(map(float, hip)), True) / scale)
        shoulder_inside = cv2.pointPolygonTest(self.bed, tuple(map(float, shoulder)), False) >= 0
        near_bed = distance >= -0.25
        seated = torso < 45 and hip_angle is not None and hip_angle < 145
        extended = hip_angle is not None and hip_angle > 150
        state, reason = State.UNKNOWN, "ambiguous posture"
        if torso > 60 and near_bed and shoulder_inside:
            state, reason = State.LYING, "horizontal torso on mattress"
        elif seated and (knee_angle is None or knee_angle < 155):
            state = State.BED_SITTING if near_bed else State.SITTING
            reason = "seated hip geometry"
        elif torso < 50 and extended and speed is not None:
            if speed > 0.25:
                state, reason = State.WALKING, "on-feet posture and displacement"
            elif speed < 0.12 and knee_angle is not None and knee_angle > 155:
                state, reason = State.STANDING, "extended legs and stationary torso"
        feet = landmarks[[27, 28]]
        feet = feet[feet[:, 2] >= 0.5, :2]
        foot_distance = None
        if len(feet):
            supporting_foot = feet[np.argmax(feet[:, 1])]
            foot_distance = float(
                cv2.pointPolygonTest(self.bed, tuple(map(float, supporting_foot)), True)
            )
        features = dict(
            torso_angle=torso,
            hip_angle=hip_angle,
            knee_angle=knee_angle,
            speed=speed,
            bed_distance=distance,
            foot_distance=foot_distance,
            torso_length=scale,
        )
        confidence = 0.8 if state != State.UNKNOWN else 0.0
        return Observation(time, state, confidence, distance, person_id, reason, features)
