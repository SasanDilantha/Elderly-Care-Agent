"""Elderly Care Agent package."""

import logging

from elderly_care_agent.config import ApplicationSettings

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ApplicationSettings"]
__version__ = "0.1.0"
