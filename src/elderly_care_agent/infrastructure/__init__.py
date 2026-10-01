"""Infrastructure adapters for files, video, and Ollama."""

from elderly_care_agent.infrastructure.json_annotation_repository import (
    JsonAnnotationRepository,
)
from elderly_care_agent.infrastructure.opencv_video_source import (
    OpenCvVideoSource,
    OpenCvVideoSourceFactory,
)

__all__ = ["JsonAnnotationRepository", "OpenCvVideoSource", "OpenCvVideoSourceFactory"]
