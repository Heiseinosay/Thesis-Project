from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class AudioFeatures:
    """Represents extracted audio features."""
    mfcc_mean: np.ndarray
    rms: np.ndarray
    zcr: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    file_name: str

@dataclass
class ProcessedAudioData:
    """Container for processed audio segment data."""
    features_dataframe: pd.DataFrame
    confidence_scores: List[float]
    overall_confidence: int
    plots: Dict[str, str]

@dataclass
class AnalysisResult:
    """Results from audio analysis operations."""
    success: bool
    data: Optional[ProcessedAudioData] = None
    error_message: Optional[str] = None
    status_code: int = 200