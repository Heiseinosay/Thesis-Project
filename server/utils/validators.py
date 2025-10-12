from dataclasses import dataclass
from flask import Request
from typing import List
from werkzeug.datastructures import FileStorage

from config.settings import AppConfig

@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str = ""
    status_code: int = 200

class AudioFileValidator:
    """Validates audio file uploads and requests."""

    ALLOWED_EXTENSIONS = [".wav", ".mp3", ".flac"]
    MAX_FILE_SIZE = 50 * 1024 * 1024 

    def validate_upload_request(self, request: Request) -> ValidationResult:
        """Validate audio upload request."""
        if 'audio_file' not in request.files:
            return ValidationResult(False, "No file part in request")
        
        file = request.files['audio_file']
        if file.filename == '':
            return ValidationResult(False, "No selected file")
        
        return self._validate_audio_file(file)
    
    def validate_record_request(self, request: Request, config: AppConfig) -> ValidationResult:
        """Validate speaker recording request."""
        if 'audio_files' not in request.files:
            return ValidationResult(False, "No files part in request")
        
        files = request.files.getlist('audio_files')

        if len(files) != config.audio_processing.required_speaker_files:
            return ValidationResult(
                False,
                f"Exactly {config.audio_processing.required_speaker_files} files required, got {len(files)}",
                400
            )
        
        for file in files:
            file_validation = self._validate_audio_files(file)
            if not file_validation.is_valid:
                return file_validation
            
        return ValidationResult(True)
    
    def _validate_audio_files(self, file: FileStorage) -> ValidationResult:
        """Validate individual audio file."""
        if not file.filename:
            return ValidationResult(False, "Empty filename")
        
        file_ext = '.' + file.filename.rsplit('.', 1)[-1].lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return ValidationResult(
                False,
                f"Unsupported file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)

        if size > self.MAX_FILE_SIZE:
            return ValidationResult(
                False,
                f"File too large, Maximum size: {self.MAX_FILE_SIZE// (1024*1024)}MB"
            )
        
        return ValidationResult(True)

