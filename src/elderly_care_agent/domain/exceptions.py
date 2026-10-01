"""Exceptions with clear boundaries between domain and input failures."""


class DomainValidationError(ValueError):
    """Raised when a domain object violates a required invariant."""


class AnnotationFormatError(ValueError):
    """Raised when an annotation file cannot be converted to domain objects."""


class DatasetPipelineError(RuntimeError):
    """Base error for an actionable dataset-pipeline failure."""


class DatasetConfigurationError(DatasetPipelineError):
    """Raised when dataset configuration is missing or inconsistent."""


class DatasetIntegrityError(DatasetPipelineError):
    """Raised when downloaded, extracted, or staged data fails validation."""


class VideoInputError(RuntimeError):
    """Raised when a video cannot be opened, decoded, or sampled safely."""
