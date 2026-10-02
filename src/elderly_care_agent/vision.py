from pathlib import Path

import cv2
import cvzone as cz
import numpy as np
from cvzone.PoseModule import PoseDetector

from ultralytics import YOLO


class Vision:
    def __init__(self, model_path, target_id=None):
        self.model = YOLO(str(model_path))
        self.pose = PoseDetector(modelComplexity=1, detectionCon=0.5, trackCon=0.5)
        self.target_id = target_id
        self.boxes = []
        self.bed_boxes = None
        self.visible_id = None

    def detect(self, frame, estimate_pose=True):
        if self.bed_boxes is None:
            beds = self.model(frame, classes=[59], conf=0.4, verbose=False)[0]
            self.bed_boxes = [
                ("bed", tuple(map(int, box.xyxy[0].tolist())), None) for box in beds.boxes
            ]
        self.visible_id = None
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.4, verbose=False
        )[0]
        self.boxes = [
            (
                self.model.names[int(box.cls[0])],
                tuple(map(int, box.xyxy[0].tolist())),
                int(box.id[0]) if box.id is not None else None,
            )
            for box in result.boxes
        ] + self.bed_boxes
        people = [(box, identity) for name, box, identity in self.boxes if name == "person"]
        if self.target_id is None and len(people) == 1:
            self.target_id = people[0][1]
        matches = [
            box for box, identity in people if identity is not None and identity == self.target_id
        ]
        if len(matches) != 1:
            return None
        self.visible_id = self.target_id
        if not estimate_pose:
            return None
        x1, y1, x2, y2 = matches[0]
        height, width = frame.shape[:2]
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            return None
        self.pose.findPose(crop, draw=False)
        result = self.pose.results.pose_landmarks
        if result is None:
            return None
        # All coordinates share image-height units to preserve joint angles.
        return np.array(
            [
                [
                    (point.x * (x2 - x1) + x1) / height,
                    (point.y * (y2 - y1) + y1) / height,
                    point.visibility,
                ]
                for point in result.landmark
            ],
            dtype=float,
        )

    def draw(self, frame, bed, state):
        for name, box, identity in self.boxes:
            x1, y1, x2, y2 = box
            cz.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), t=2)
            cz.putTextRect(frame, f"{name} {identity or ''}", (max(0, x1), max(25, y1)), scale=1)
        cv2.polylines(frame, [bed.astype(np.int32)], True, (0, 255, 255), 2)
        cz.putTextRect(frame, f"{state.upper()} | {state.bed.upper()}", (15, 35), scale=1)
        return frame

    def close(self):
        self.pose.pose.close()

    @staticmethod
    def bed_region(frame, rectangle=None, select=False):
        height, width = frame.shape[:2]
        if select:
            x, y, w, h = cv2.selectROI("Select mattress", frame, fromCenter=False)
            cv2.destroyWindow("Select mattress")
            rectangle = [x / width, y / height, (x + w) / width, (y + h) / height]
        if rectangle is None:
            raise ValueError("Supply --bed left,top,right,bottom or --select-bed.")
        left, top, right, bottom = rectangle
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("Bed coordinates must form a nonempty rectangle within [0, 1].")
        return np.array(
            [
                [left * width, top * height],
                [right * width, top * height],
                [right * width, bottom * height],
                [left * width, bottom * height],
            ]
        )

    @staticmethod
    def default_model():
        directory = Path("data/cache/vision")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "yolo26s.pt"
