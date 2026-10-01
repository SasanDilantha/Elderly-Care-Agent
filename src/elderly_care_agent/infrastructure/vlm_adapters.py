"""OpenCV JPEG encoding and local Ollama vision requests."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from ollama import Client

from elderly_care_agent.application.vlm_ports import ContextFrameEncoder, VisionLanguageModel
from elderly_care_agent.config import VlmSettings
from elderly_care_agent.domain.exceptions import VlmServiceError
from elderly_care_agent.infrastructure.opencv_video_source import ImageFrame

logger = logging.getLogger(__name__)


class OpenCvJpegEncoder(ContextFrameEncoder[ImageFrame]):
    """Compresses a frame in memory without creating image files."""

    def __init__(self, max_edge: int = 512, quality: int = 85) -> None:
        if max_edge <= 0 or not 1 <= quality <= 100:
            raise ValueError("invalid JPEG encoder dimensions or quality")
        self._max_edge = max_edge
        self._quality = quality

    def encode(self, frame: ImageFrame) -> bytes:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise VlmServiceError("context frame must be an 8-bit BGR image")
        height, width = frame.shape[:2]
        if max(height, width) > self._max_edge:
            scale = self._max_edge / max(height, width)
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not encoded:
            raise VlmServiceError("cannot encode context frame as JPEG")
        return buffer.tobytes()


class OllamaVisionLanguageModel(VisionLanguageModel):
    """Keeps the Ollama SDK and its network failures behind one adapter."""

    def __init__(self, settings: VlmSettings, client: Client | None = None) -> None:
        self._settings = settings
        self._client = client or Client(host=settings.base_url, timeout=settings.timeout_sec)

    def complete(self, prompt: str, images: Sequence[bytes], schema: dict) -> str:
        if not images:
            raise VlmServiceError("at least one context image is required")
        logger.info(
            "Ollama request started | model=%s | images=%d | endpoint=%s",
            self._settings.model,
            len(images),
            self._settings.base_url,
        )
        try:
            response = self._client.chat(
                model=self._settings.model,
                messages=[{"role": "user", "content": prompt, "images": list(images)}],
                format=schema,
                options={
                    "temperature": self._settings.temperature,
                    "num_ctx": self._settings.context_window_tokens,
                },
                stream=False,
            )
            content = response.message.content
        except Exception as error:
            raise VlmServiceError(
                f"local Ollama request failed for {self._settings.model}: {error}"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise VlmServiceError("local Ollama returned an empty response")
        logger.info("Ollama request completed | model=%s", self._settings.model)
        return content
