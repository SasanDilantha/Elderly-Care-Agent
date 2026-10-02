import json
from pathlib import Path

import cv2
import numpy as np

from elderly_care_agent.pipeline import CareMonitor


def main():
    source = Path("data/raw/gmdcsa24/development/Subject 1/05.mp4")
    directory = Path("outputs/robustness")
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in ["poor_lighting", "temporary_occlusion", "second_person_proxy"]:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            raise ValueError("Prepare the dataset first")
        path = directory / f"{case}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
        if not writer.isOpened():
            raise RuntimeError("Cannot create robustness video")
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (640, 360))
                if case == "poor_lighting":
                    frame = (frame.astype(float) * 0.1).astype(np.uint8)
                elif case == "temporary_occlusion" and 5 <= index / fps <= 7:
                    frame[:] = 0
                elif case == "second_person_proxy":
                    left = cv2.resize(frame, (320, 360))
                    frame = np.hstack([left, cv2.flip(left, 1)])
                writer.write(frame)
                index += 1
        finally:
            cap.release()
            writer.release()
        report = CareMonitor().run(path)
        CareMonitor.save(report, directory / case)
        results[case] = {
            key: report[key] for key in ["activity_duration_sec", "events", "decision"]
        }
    (directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
