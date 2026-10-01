"""MediaPipe Tasks adapter for timestamped pose estimation."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from elderly_care_agent.application.vision_ports import PoseEstimator, PoseEstimatorFactory
from elderly_care_agent.domain.exceptions import VisionModelError
from elderly_care_agent.domain.vision import PoseLandmark
from elderly_care_agent.infrastructure.opencv_video_source import ImageFrame

logger = logging.getLogger(__name__)


class MediaPipePoseEstimator(PoseEstimator[ImageFrame]):
    """Keeps one video-mode task open across sampled frames."""

    def __init__(
        self, model_path: Path, detection_confidence: float, tracking_confidence: float
    ) -> None:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        try:
            self._task = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as error:
            raise VisionModelError(f"cannot load pose model {model_path}: {error}") from error
        logger.info("MediaPipe pose estimator opened | model=%s", model_path)

    def estimate(self, frame: ImageFrame, timestamp_sec: float) -> tuple[PoseLandmark, ...]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise VisionModelError("pose input must be an 8-bit BGR image")
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        )
        try:
            result = self._task.detect_for_video(image, round(timestamp_sec * 1000))
        except (RuntimeError, ValueError) as error:
            raise VisionModelError(
                f"pose detection failed at {timestamp_sec:.3f}s: {error}"
            ) from error
        if not result.pose_landmarks:
            return ()
        return tuple(
            PoseLandmark(
                index=index,
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(landmark.visibility),
                presence=float(landmark.presence),
            )
            for index, landmark in enumerate(result.pose_landmarks[0])
        )

    def close(self) -> None:
        self._task.close()
        logger.info("MediaPipe pose estimator closed")


class MediaPipePoseEstimatorFactory(PoseEstimatorFactory[ImageFrame]):
    def create(
        self,
        model_path: Path,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> MediaPipePoseEstimator:
        return MediaPipePoseEstimator(model_path, detection_confidence, tracking_confidence)
