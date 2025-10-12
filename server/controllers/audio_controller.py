from flask import Blueprint, request, jsonify, Response
from werkzeug.datastructures import FileStorage
from typing import List
import json

from services.model_service import ModelService
from services.audio_processor import AudioProcessor
from services.visualization import VisualizationService
from models.audio_data import AnalysisResult, ProcessedAudioData
from exceptions.audio_exceptions import AudioProcessingError, ModelLoadingError, InsufficientDataError
from utils.validators import AudioFileValidator

class AudioController:
    """Handles HTTP requests for audio analysis operations."""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.audio_processor = AudioProcessor(model_service.config)
        self.visualization_service = VisualizationService()
        self.validator = AudioFileValidator()
        
        # Store processed data
        self._uploaded_data = None
        self._speaker_data = None
        
        # Create blueprint
        self.blueprint = Blueprint('audio', __name__, url_prefix='/api')
        self._register_routes()
    
    def _register_routes(self):
        """Register all route handlers."""
        self.blueprint.add_url_rule('/greet/tie', 'greetings', self.greetings, methods=['GET'])
        self.blueprint.add_url_rule('/upload', 'upload_audio', self.upload_audio, methods=['POST'])
        self.blueprint.add_url_rule('/record', 'record_audio', self.record_audio, methods=['POST'])
    
    def greetings(self):
        """Health check endpoint."""
        return jsonify({"Hello": "World"})
    
    def upload_audio(self):
        """Handle audio file upload and deepfake analysis."""
        try:
            # Validate request
            validation_result = self.validator.validate_upload_request(request)
            if not validation_result.is_valid:
                return jsonify({"message": validation_result.error_message}), 400
            
            # Process audio
            file = request.files['audio_file']
            features = self.audio_processor.process_audio_segments([file])
            segment_df = self.audio_processor.create_features_dataframe(features)
            
            # Analyze with deepfake model
            confidence_scores = self.model_service.deepfake_segments(segment_df)
            overall_confidence = self.model_service.average(confidence_scores)
            
            # Generate visualizations
            plots = self.visualization_service.generate_feature_plots(segment_df)
            
            # Store processed data
            self._uploaded_data = segment_df
            
            # Prepare response
            processed_data = ProcessedAudioData(
                features_dataframe=segment_df,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                plots=plots
            )
            
            result = AnalysisResult(success=True, data=processed_data)
            return self._create_response(result)
            
        except AudioProcessingError as e:
            return self._handle_error(str(e), 400)
        except ModelLoadingError as e:
            return self._handle_error(str(e), 500)
        except Exception as e:
            return self._handle_error(f"Unexpected error: {str(e)}", 500)
    
    def record_audio(self):
        """Handle speaker voice recording and identification."""
        try:
            # Validate request
            validation_result = self.validator.validate_record_request(request, self.model_service.config)
            if not validation_result.is_valid:
                return jsonify({"message": validation_result.error_message}), validation_result.status_code
            
            # Check if uploaded data exists
            if self._uploaded_data is None:
                return jsonify({"message": "No uploaded audio data found"}), 400
            
            # Process speaker files
            files = request.files.getlist('audio_files')
            
            # Train model if needed
            if not self.model_service.has_speaker_model or len(files) == self.model_service.config.REQUIRED_SPEAKER_FILES:
                features = self.audio_processor.process_audio_segments(files)
                speaker_df = self.audio_processor.create_features_dataframe(features)
                self._speaker_data = speaker_df
                
                # Train new model
                self.model_service.train_speaker_model(speaker_df)
            
            # Analyze uploaded audio with speaker model
            confidence_scores = self.model_service.speaker_segments(self._uploaded_data)
            overall_confidence = self.model_service.average(confidence_scores)
            
            # Generate comparison plots
            plots = self.visualization_service.generate_feature_plots(
                self._uploaded_data, self._speaker_data
            )
            
            # Prepare response
            processed_data = ProcessedAudioData(
                features_dataframe=self._speaker_data,
                confidence_scores=confidence_scores,
                overall_confidence=overall_confidence,
                plots=plots
            )
            
            result = AnalysisResult(success=True, data=processed_data)
            return self._create_response(result)
            
        except InsufficientDataError as e:
            return self._handle_error(str(e), 400)
        except AudioProcessingError as e:
            return self._handle_error(str(e), 400)
        except ModelLoadingError as e:
            return self._handle_error(str(e), 500)
        except Exception as e:
            return self._handle_error(f"Unexpected error: {str(e)}", 500)
    
    def _create_response(self, result: AnalysisResult) -> Response:
        """Create standardized JSON response."""
        if not result.success:
            return jsonify({"error": result.error_message}), result.status_code
        
        response_data = {
            "overall": result.data.overall_confidence,
            **result.data.plots,
            "uploaded_data": result.data.features_dataframe.to_dict(orient="records")
        }
        
        return Response(
            json.dumps(response_data),
            mimetype='application/json',
            status=200
        )
    
    def _handle_error(self, message: str, status_code: int) -> Response:
        """Handle error responses consistently."""
        return jsonify({"error": message}), status_code