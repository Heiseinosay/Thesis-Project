import librosa
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from werkzeug.datastructures import FileStorage

from models.audio_data import AudioFeatures
from config.settings import AppConfig
from server.exceptions.audio_exceptions import AudioProcessingError, ModelLoadingError, InsufficientDataError

class AudioProcessor:
    """Handles all audio preprocessing and feature extraction operations."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.target_sr = config.audio_processing.target_sample_rate
        self.segment_duration = config.audio_processing.segment_duration
        self.frame_length = config.audio_processing.frame_length
        self.hop_length = config.audio_processing.hop_length
        self.n_mfcc = config.audio_processing.n_mfcc
        self.required_speaker_files = config.audio_processing.required_speaker_files
        self.res_type = config.audio_processing.res_type

    def preprocess_audio(self, audio_file: FileStorage) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file to standardized format.

        Args:
            audio_file: Uploaded audio file

        Returns:
            signal (np.ndarray): Mono audio signal
            sr (int): Sample rate
        """
        if not isinstance(audio_file, FileStorage):
            raise TypeError("Audio file must be a FileStorage object.")

        try:
            signal, sr = librosa.load(
                audio_file.stream,
                sr=self.target_sr,
                mono=False,
                res_type=self.res_type
            )

            if signal.size == 0:
                raise ValueError("Loaded audio signal is empty.")

            return signal, sr
        
        except Exception as e:
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
    
    def extract_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """Comput MFCC Features"""

        if signal.size == 0:
            raise ValueError("Signal is empty.")
        
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc
        )

        return mfcc
    
    def extract_delta_mfcc(self, mfcc: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute delta MFCC features"""

        return librosa.feature.delta(mfcc=mfcc,order=order)
    
    def extract_mfcc_features(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Concat MFCC + delta MFCC + delta-delta MFCC and concatenate along feature axis."""
        
        mfcc = self.extract_mfcc(signal)
        delta = self.extract_delta_mfcc(mfcc, order=1)
        delta2 = self.extract_delta_mfcc(mfcc, order=2)
        return np.concatenate((mfcc, delta, delta2), axis = 0)

    def extract_rms(self, signal: np.ndarray) -> np.ndarray:
        """Get a 1D array of rms value for each frame """

        rms = librosa.feature.rms(
            y=signal,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return rms
    
    def extract_zcr(self, signal: np.ndarray) -> np.ndarray:
        """Get the 1D array of zcr value for each frame"""

        zcr = librosa.feature.rms(
            y=signal,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return zcr
    
    def extract_sc(self, signal: np.ndarray) -> np.ndarray:
        """Get the 1D array of spectral centroid value for each frame"""
        spectral_centroid = librosa.feature.spectral_centroid(
            y=signal,
            sr=self.target_sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return spectral_centroid

    def extract_sb(self, signal: np.ndarray) -> np.ndarray:
        """Get the 1D array of spectral bandwidth value for each frame"""
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal,
            sr=self.target_sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return spectral_bandwidth

    def _process_segment(self, segment: np.ndarray, segment_number: int, file_name: str) -> AudioFeatures:
        """Process a single segment and return AudioFeatures object."""

        mfcc = self.extract_mfcc_features(signal = segment)
        mfcc_mean = np.mean(mfcc.T, axis = 0)
        rms = self.extract_rms(signal = segment)
        zcr = self.extract_zcr(signal=segment)
        sc = self.extract_sc(signal=segment)

        sb = self.extract_sb(signal=segment)

        return AudioFeatures(
            mfcc_mean = mfcc_mean,
            rms = rms,
            zcr = zcr,
            spectral_centroid= sc,
            spectral_bandwidth= sb,
            file_name = f"Segment {segment_number} {file_name}"
        )
    
    def process_audio_segments(self, files: List[FileStorage]) -> List[AudioFeatures]:
        all_features = []

        for audio_file in files:
            try:
                signal, sr = self.preprocess_audio(audio_file=audio_file)
                segment_length = int(self.segment_duration*sr)
                total_length = len(signal)

                segment_number = 0
                for start in range(0, total_length, segment_length):
                    segment_number += 1
                    end = start + segment_length
                    segment = signal[start:end]

                    # Pad last segment if shorter
                    if len(segment) < segment_length:
                        segment = np.pad(segment, 
                                         (0, segment_length - len(segment)), 
                                         mode = "constant")

                    features = self._process_segment(segment=segment,
                                                     segment_number=segment_number,
                                                     audio_file=audio_file.filename)
                    all_features.append(features)
            except Exception as e:
                raise AudioProcessingError(f"Failed to process {audio_file.filename}: {str(e)}")
        return all_features
    
    def create_features_dataframe(self, features_list: List[AudioFeatures]) -> pd.DataFrame:
        """Convert list of AudioFeatures to structured DataFrame."""
        if not features_list:
            return pd.DataFrame()
        
        df = pd.DataFrame({'File Name' : [f.file_name for f in features_list]})

        # Add MFCC columns
        mfcc_data = np.stack([f.mfcc for f in features_list])
        mfcc_cols = [f"MFCC{i+1}" for i in range(mfcc_data.shape[1])]
        df[mfcc_cols] = mfcc_data

        for feat_name in ["rms", "zcr", "spectral_centroid", "spectral_bandwidth"]:
            feat_data = [getattr(f, feat_name) for f in features_list]

            # Pad sequences to max length
            max_len = max(len(fd) for fd in feat_data)
            padded = np.array([np.pad(fd, (0, max_len - len(fd)), mode="constant") for fd in feat_data])

            col_names = [f"{feat_name.capitalize()}{i+1}" for i in range(max_len)]
            df[col_names] = padded
        
        return df