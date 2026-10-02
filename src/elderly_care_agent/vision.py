from pathlib import Path

import cv2
import cvzone as cz
import numpy as np
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO


class Vision:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path))
        self.bed_model = YOLO(str(model_path))
        self.pose = PoseDetector(modelComplexity=1, detectionCon=0.5, trackCon=0.5)
        self.target_id = None
        self.boxes = []
        self.bed_history = []
        self.bed_box = None
        self.bed_polygon = None
        self.next_bed_scan = 0.0
        self.visible_id = None

    def detect(self, frame, time=0.0, estimate_pose=True):
        if estimate_pose and time >= self.next_bed_scan and len(self.bed_history) < 5:
            self.locate_bed(frame)
            self.next_bed_scan = time + 1.0
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
        ]
        if self.bed_box is not None:
            self.boxes.append(("bed", self.bed_box, None))
        people = [(box, identity) for name, box, identity in self.boxes if name == "person"]
        if self.target_id is None:
            self.select_person(people)
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

    def locate_bed(self, frame):
        result = self.bed_model(frame, classes=[59], conf=0.4, verbose=False)[0]
        candidates = [box.xyxy[0].tolist() for box in result.boxes]
        if len(candidates) != 1:
            return
        box = candidates[0]
        if self.bed_box is not None and self.overlap(box, self.bed_box) < 0.5:
            return
        self.bed_history.append(box)
        self.bed_box = tuple(map(int, np.median(self.bed_history, axis=0)))
        left, top, right, bottom = self.bed_box
        # The upper half estimates mattress contact; YOLO detects the whole bed.
        surface_bottom = top + 0.5 * (bottom - top)
        self.bed_polygon = np.array(
            [[left, top], [right, top], [right, surface_bottom], [left, surface_bottom]]
        )

    @staticmethod
    def overlap(first, second):
        left, top = max(first[0], second[0]), max(first[1], second[1])
        right, bottom = min(first[2], second[2]), min(first[3], second[3])
        area = max(0, right - left) * max(0, bottom - top)
        return area / max(1, (first[2] - first[0]) * (first[3] - first[1]))

    def select_person(self, people):
        tracked = [(box, identity) for box, identity in people if identity is not None]
        if len(people) == 1 and tracked:
            self.target_id = tracked[0][1]
        elif self.bed_box is not None and len(tracked) == len(people) and tracked:
            scores = sorted(
                [(self.overlap(box, self.bed_box), identity) for box, identity in tracked],
                reverse=True,
            )
            if scores[0][0] >= 0.25 and scores[0][0] - scores[1][0] >= 0.2:
                self.target_id = scores[0][1]

    def draw(self, frame, state):
        for name, box, identity in self.boxes:
            x1, y1, x2, y2 = box
            cz.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), t=2)
            cz.putTextRect(
                frame, f"{name} {identity or ''}", (max(10, x1), max(25, y1)), scale=1, thickness=1
            )
        if self.bed_polygon is not None:
            cv2.polylines(frame, [self.bed_polygon.astype(np.int32)], True, (0, 255, 255), 2)
        cz.putTextRect(
            frame, f"{state.upper()} | {state.bed.upper()}", (15, 35), scale=1, thickness=1
        )
        return frame

    def close(self):
        self.pose.pose.close()

    @staticmethod
    def default_model():
        directory = Path("data/cache/vision")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "yolo26s.pt"
