import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from werkzeug.datastructures import FileStorage
import io

from services.audio_processor import AudioProcessor
from models.audio_data import AudioFeatures
from config.settings import AppConfig
from exceptions.audio_exceptions import AudioProcessingError

class TestAudioProcessor(unittest.TestCase):
    """Test suite for AudioProcessor class."""

    def setUp(self):
        """Set up test fixtures."""

        self.config = AppConfig()
        self.processor = AudioProcessor(self.config)

    def test_preprocess_audio_mono(self):
        """Test audio preprocesssing with mono audio."""

        # Create mock audio file
        mock_file = Mock(spec=FileStorage)
        mock_signal = np.array([0.1, 0.2, 0.3, 0.4])

        with patch('librosa.load') as mock_load:
            mock_load.return_value = (mock_signal, 22050)

            signal, sr = self.processor.preprocess_audio(mock_file)

            self.assertEqual(sr, 22050)
            np.testing.assert_array_equal(signal, mock_signal)
    
    def test_preprocess_audio_stereo(self):
        """Test audio preprocessomg with stereo audio."""

        mock_file = Mock(spec=FileStorage)
        mock_signal = np.array([[0.1, 0.2], [0.3, 0.4]])

        with patch('librosa.load') as mock_load:
            mock_load.return_value = (mock_load, 22050)

            signal, sr = self.processor.preprocess_audio(mock_file)

            expected = np.mean(mock_signal, axis = 0)
            np.testing.assert_array_equal(signal, expected)

    # TODO: Add more tests