"""Replaceable image encoding and local VLM request ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class ContextFrameEncoder[FramePayload](ABC):
    @abstractmethod
    def encode(self, frame: FramePayload) -> bytes:
        """Encode one frame as a compact JPEG image."""


class VisionLanguageModel(ABC):
    @abstractmethod
    def complete(self, prompt: str, images: Sequence[bytes], schema: dict) -> str:
        """Return one structured response for ordered context images."""
