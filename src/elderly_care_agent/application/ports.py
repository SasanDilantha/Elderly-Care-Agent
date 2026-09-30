"""Ports implemented by infrastructure in later pipeline layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from elderly_care_agent.domain.models import GroundTruthAnnotation


class AnnotationRepository(ABC):
    """Loads annotations without exposing their storage format to use cases."""

    @abstractmethod
    def load(self, path: Path) -> GroundTruthAnnotation:
        """Load and validate one annotation file."""
