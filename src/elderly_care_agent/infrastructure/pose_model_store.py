"""Pinned official MediaPipe pose model in the ignored project cache."""

from __future__ import annotations

import logging
from pathlib import Path

from elderly_care_agent.application.dataset_ports import DatasetDownloader
from elderly_care_agent.application.vision_ports import PoseModelProvider
from elderly_care_agent.domain.exceptions import DatasetPipelineError, VisionModelError
from elderly_care_agent.infrastructure.dataset_io import FileHasher, HttpDatasetDownloader

logger = logging.getLogger(__name__)


class PoseModelStore(PoseModelProvider):
    SOURCE_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"

    def __init__(
        self,
        path: Path = Path("data/cache/vision/pose_landmarker_lite.task"),
        downloader: DatasetDownloader | None = None,
    ) -> None:
        self._path = path
        self._downloader = downloader or HttpDatasetDownloader()

    def ensure_available(self) -> Path:
        if self._path.is_file():
            if FileHasher.digest(self._path) == self.SHA256:
                logger.info("Pose model ready | path=%s | sha256=%s", self._path, self.SHA256)
                return self._path.resolve()
            logger.warning("Removing invalid cached pose model | path=%s", self._path)
            self._path.unlink()

        logger.info("Pose model download started | source=%s", self.SOURCE_URL)
        try:
            self._downloader.download(self.SOURCE_URL, self._path)
            actual = FileHasher.digest(self._path)
            if actual != self.SHA256:
                raise VisionModelError(
                    f"pose model checksum mismatch: expected {self.SHA256}, got {actual}"
                )
        except (OSError, DatasetPipelineError, VisionModelError) as error:
            self._path.unlink(missing_ok=True)
            raise VisionModelError(f"cannot prepare pose model: {error}") from error
        logger.info("Pose model verified | path=%s | sha256=%s", self._path, self.SHA256)
        return self._path.resolve()
