import logging
from flask import Flask

def setup_logger(app: Flask):
    """Configure application logging."""
    if not app.debug:
        # Production logging configuration
        handler = logging.FileHandler('audio_analysis.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Audio Analysis Service startup')