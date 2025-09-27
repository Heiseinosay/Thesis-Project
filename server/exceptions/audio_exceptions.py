class AudioProcessingError(Exception):
    """Raised when audio processing operations fail."""
    pass

class ModelLoadingError(Exception):
    """Raised when model loading fails."""
    pass

class InsufficientDataError(Exception):
    """Raised when insufficient data is provided for analysis."""
    pass