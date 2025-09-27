import os
import yaml
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List, Dict

class SettingsConfig(BaseModel):
    debug_mode: bool = False
    host: str = "0.0.0.0"
    port: str = 8080

    @property
    def flask_config(self) -> Dict:
        return {
            "DEBUG": self.debug_mode,
            "HOST": self.host,
            "PORT": self.port
        }
    
class AudioConfig(BaseModel):
    target_sample_rate: int
    segment_duration: int
    frame_length: int
    hop_length: int
    n_mfcc: int
    required_speaker_files: int
    res_type: str
    audio_extensions: List[str]
    max_file_size: int

class PathConfig(BaseModel):
    # Path
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / 'models'
    scalers_dir = base_dir / 'scalers'
    results_dir = base_dir / 'results'

    # Model files
    deepfake_model = models_dir / 'DeepFake_model_ver5_full.keras'
    deepfake_scaler = scalers_dir / 'df_scaler.pkl'
    speaker_scaler = scalers_dir / 'si_scaler.pkl'
    vr_data = results_dir / 'voice_recognition' / 'training_full' / 'vr_other_segment.csv'

class AppConfig(BaseModel):
    """Base configuration class"""

    settings: SettingsConfig
    audio_processing: AudioConfig
    path: PathConfig

    # CORS Settings
    CORS_ORIGINS = os.getenv('CORS_ORIGIN', '*')

    # DEBUG = bool(os.getenv('DEBUG'))
    # HOST = os.getenv('HOST')
    # PORT = int(os.getenv('PORT'))

    @classmethod
    def load_config(cls, yaml_path: Path) -> "AppConfig":
        try:
            with open(yaml_path, 'r') as file:
                config_data = yaml.safe_load(file)

            data = cls(**config_data)

            if data.settings.debug_mode:
                print("Application is running in DEBUG MODE")

            return data
        
        except FileNotFoundError:
            raise f""
        except ValidationError:
            print("input logger")
        except yaml.YAMLError as e:
            print("input logger")

# class DevelopmentConfig(AppConfig):
#     DEBUG = True

# class ProductionConfig(AppConfig):
#     DEBUG = False

ENV = os.getenv("ENV", "development")
config_file = Path(__file__).resolve().parent.parent
config = AppConfig.load_config(config_file)