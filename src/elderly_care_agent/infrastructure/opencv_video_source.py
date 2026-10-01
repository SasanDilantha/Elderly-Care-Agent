"""OpenCV implementation of the video input port."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from elderly_care_agent.application.video_ports import VideoSource, VideoSourceFactory
from elderly_care_agent.domain.exceptions import VideoInputError
from elderly_care_agent.domain.video import TimestampedFrame, VideoMetadata

logger = logging.getLogger(__name__)
ImageFrame = NDArray[np.uint8]


class OpenCvVideoSource(VideoSource[ImageFrame]):
    """Reads validated metadata and exact frame indices sequentially."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._metadata = self._read_metadata()

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def read_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[TimestampedFrame[ImageFrame]]:
        indices = tuple(frame_indices)
        self._validate_indices(indices)
        if not indices:
            return

        logger.info(
            "Video decoding started | path=%s | requested_frames=%d",
            self._path,
            len(indices),
        )
        capture = self._open_capture()
        target_position = 0
        frame_index = 0
        try:
            while target_position < len(indices):
                readable, image = capture.read()
                if not readable:
                    raise VideoInputError(
                        f"video ended before frame {indices[target_position]}: {self._path}"
                    )
                if frame_index == indices[target_position]:
                    timestamp_sec = frame_index / self._metadata.fps
                    logger.debug(
                        "Decoded sampled frame | index=%d | timestamp_sec=%.6f",
                        frame_index,
                        timestamp_sec,
                    )
                    yield TimestampedFrame(frame_index, timestamp_sec, image)
                    target_position += 1
                frame_index += 1
        finally:
            capture.release()
        logger.info("Video decoding completed | decoded_samples=%d", len(indices))

    def _read_metadata(self) -> VideoMetadata:
        if not self._path.is_file():
            raise VideoInputError(f"video file does not exist: {self._path}")
        capture = self._open_capture()
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise VideoInputError(f"invalid or unsupported video metadata: {self._path}")
        metadata = VideoMetadata(
            video_id=self._path.name,
            source_path=str(self._path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        logger.info(
            "Video metadata loaded | path=%s | fps=%.3f | frames=%d | size=%dx%d | duration=%.3f",
            self._path,
            metadata.fps,
            metadata.frame_count,
            metadata.width,
            metadata.height,
            metadata.duration_sec,
        )
        return metadata

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"OpenCV cannot open video: {self._path}")
        return capture

    def _validate_indices(self, indices: tuple[int, ...]) -> None:
        if any(index < 0 or index >= self._metadata.frame_count for index in indices):
            raise VideoInputError("requested frame index is outside the video")
        if any(
            current <= previous for previous, current in zip(indices, indices[1:], strict=False)
        ):
            raise VideoInputError("frame indices must be strictly increasing")


class OpenCvVideoSourceFactory(VideoSourceFactory[ImageFrame]):
    """Creates OpenCV sources for application services."""

    def create(self, path: Path) -> OpenCvVideoSource:
        return OpenCvVideoSource(path)
