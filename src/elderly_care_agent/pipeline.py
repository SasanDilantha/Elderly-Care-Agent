import math
from collections import defaultdict
from dataclasses import asdict

import cv2
import numpy as np

from elderly_care_agent.activity import ActivityRules
from elderly_care_agent.agent import ContextAgent
from elderly_care_agent.events import AlertPolicy, BedEvents
from elderly_care_agent.models import Settings, State
from elderly_care_agent.report import Report
from elderly_care_agent.timeline import Timeline
from elderly_care_agent.vision import Vision


class CareMonitor:
    def __init__(self, settings=None, model=None):
        self.settings = settings or Settings()
        self.model = model or Vision.default_model()

    def run(self, video, show=False, use_vlm=False):
        cap = cv2.VideoCapture(str(video))
        vision = None
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0 or self.settings.sample_fps <= 0:
                raise ValueError("Video FPS and sample FPS must be positive.")
            success, frame = cap.read()
            if not success:
                raise ValueError("No readable frames")
            rules = ActivityRules()
            vision = Vision(self.model)
            observations, index, next_sample = [], 0, 0.0
            completed = True
            while success:
                time = index / fps
                sample = time + 1e-8 >= next_sample
                landmarks = vision.detect(frame, time, estimate_pose=sample)
                if sample:
                    if vision.bed_polygon is not None:
                        rules.bed = (vision.bed_polygon / frame.shape[0]).astype(np.float32)
                    row = rules.classify(time, landmarks, vision.visible_id)
                    observations.append(row)
                    next_sample += 1 / min(fps, self.settings.sample_fps)
                    if show:
                        cv2.imshow("Elderly Care", vision.draw(frame.copy(), row.state))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            completed = False
                            index += 1
                            break
                index += 1
                success, frame = cap.read()
            duration = index / fps
            segments = Timeline(self.settings).build(observations, duration)
            agent = ContextAgent(self.settings, use_vlm)
            segments = agent.review(video, segments, observations)
            events = BedEvents(self.settings).detect(segments, observations)
            report = self.summarize(segments, events, duration)
            report.update(
                video=str(video),
                completed=completed,
                decoded_frames=index,
                sampled_frames=len(observations),
                fps=fps,
                settings=asdict(self.settings),
                bed_polygon=vision.bed_polygon.tolist() if vision.bed_polygon is not None else None,
                scene_setup=dict(
                    mode="automatic",
                    bed_box=vision.bed_box,
                    bed_detections=len(vision.bed_history),
                    contact_region="upper half of detected bed; approximate mattress surface",
                    target_id=vision.target_id,
                ),
                agent_actions=agent.actions,
                observations=[asdict(row) for row in observations],
            )
            report["decision"] = AlertPolicy(self.settings).decide(segments, events)
            return report
        finally:
            cap.release()
            if vision:
                vision.close()
            if show:
                cv2.destroyAllWindows()

    @staticmethod
    def summarize(segments, events, duration):
        activity = {state.value: 0.0 for state in State}
        occupancy = defaultdict(float)
        longest, outside = 0.0, 0.0
        for row in segments:
            activity[row.state.value] += row.duration
            occupancy[row.state.bed] += row.duration
            outside = outside + row.duration if row.state.bed == "out_of_bed" else 0.0
            longest = max(longest, outside)
        return dict(
            observation_duration_sec=duration,
            activity_duration_sec=activity,
            total_in_bed_sec=occupancy["in_bed"],
            total_out_of_bed_sec=occupancy["out_of_bed"],
            total_bed_unknown_sec=occupancy["unknown"],
            longest_out_of_bed_period_sec=longest,
            bed_exit_count=sum(e["event"] == "bed_exit" for e in events),
            bed_return_count=sum(e["event"] == "bed_return" for e in events),
            final_state=segments[-1].state if segments else State.UNKNOWN,
            timeline=[row.to_dict() for row in segments],
            events=events,
        )

    @staticmethod
    def save(report, directory):
        return Report(report).save(directory)
