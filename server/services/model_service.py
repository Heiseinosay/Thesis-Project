import joblib
import numpy as np
import pandas as pd
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Any
import math

from config.settings import AppConfig
from exceptions.audio_exceptions import AudioProcessingError, ModelLoadingError, InsufficientDataError

class ModelService:
    """Manages machine learning models and their operations."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._deepfake_model = None
        self._speaker_model = None
        self._df_scaler = None
        self._si_scaler = None
        self._others_profile = None

        self._load_models()

    def _load_models(self):
        """Load pre-trained models and scalers."""

        try:
            self._deepfake_model = load_model(str(self.config.path.deepfake_model))
            self._df_scaler = joblib.load(str(self.config.path.deepfake_scaler))
            self._si_scaler = joblib.load(str(self.config.path.speaker_scaler))
            self._others_profile = pd.read_csv(str(self.config.path.vr_data))
        except Exception as e:
            raise ModelLoadingError(f"Failed to load models: {str(e)}")
        
    def _reshape_segment(self, segment_features: np.ndarray, scaler) -> np.ndarray:
        """Reshape segment data for model input."""
        reshaped = scaler.transform([segment_features])
        return reshaped.reshape(1, reshaped.shape[1], 1, 1)
    
    def _evaluate_segments(self, segment_data: pd.DataFrame, model, scaler) -> List[float]:
        """
        Segment evaluation

        Args:
            segment_data: DataFrame containing segment features
        
        Returns:
            List of confidence scores
        """

        if model is None:
            raise ModelLoadingError("Model not loaded")
        
        features = segment_data.drop(columns=['File Name']).values
        confidence_scores = []

        for segment_row in features:
            reshaped_segment = self._reshape_segment(segment_row, scaler)

            predictions = model.predict(reshaped_segment, verbose=0)
            confidence = float(predictions[0][0])
            confidence_scores.append(confidence)

        return confidence_scores
    
    def deepfake_segments(self, segment_data:pd.DataFrame) -> List[float]:
        return self._evaluate_segments(segment_data, self._deepfake_model, self._df_scaler)
    
    def speaker_segments(self, segment_data:pd.DataFrame) -> List[float]:
        return self._evaluate_segments(segment_data, self._speaker_model, self._si_scaler)
     
    def _reshape_data(self, data: np.ndarray) -> np.ndarray[tuple[int, int, int, int], Any]:
        return data.reshape(data.shape[0], data.shape[1], 1, 1)
    
    def train_speaker_model(self, speaker_data: pd.DataFrame) -> None:
        """
        Train a new speaker identification model.

        Args:
            speaker_data: DataFrame containing speaker voice feature
        """

        if len(speaker_data) < self.config.audio_processing.required_speaker_files:
            raise InsufficientDataError(
                f"Requires {self.config.audio_processing.required_speaker_files} files, got {len(speaker_data)}"
            )
        
        speaker_profile = speaker_data.copy()
        speaker_profile['label'] = 1

        others_profile = self._others_profile.copy()
        others_profile['label'] = 0
        others_profile.columns = speaker_profile.columns

        combined_data = pd.concat([speaker_profile, others_profile])
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        X = combined_data.drop(columns=['File Name', 'label']).values
        y = combined_data['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = self._reshape_data(X_train)
        X_test = self._reshape_data(X_test)

        # Build Model
        input_shape = (X_train.shape[1], 1, 1)
        model = self._build_model(input_shape)

        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_split=0.01,
            verbose=0
        )

        model.evaluate(X_test, y_test, verbose=0)
        self._speaker_model = model

    def _build_speaker_model(self, input_shape: Any):
        """Build CNN model for speaker identification"""
        model = Sequential([
            Conv2D(16,(3, 3), activation='relu', input_shape=input_shape, padding='same'),
            MaxPooling2D((2,1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2,1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer="adam",
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    @staticmethod
    def average(confidence_scores : List[float]) -> int:
        """Calucalte overall confidence from individual scores."""
        if not confidence_scores:
            return 0
        
        avg = sum(confidence_scores)/len(confidence_scores)
        rounded = math.ceil(avg*100)
        return rounded
    
    @property
    def has_speaker_model(self) -> bool:
        """Check if speaker identification model is available."""
        return self._speaker_model is not None
    