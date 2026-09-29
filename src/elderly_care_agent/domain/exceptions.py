"""Exceptions with clear boundaries between domain and input failures."""


class DomainValidationError(ValueError):
    """Raised when a domain object violates a required invariant."""


class AnnotationFormatError(ValueError):
    """Raised when an annotation file cannot be converted to domain objects."""

