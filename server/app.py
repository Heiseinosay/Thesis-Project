from flask import Flask
from flask_cors import CORS
from config.settings import config, AppConfig

from utils.logger import setup_logger
from services.model_service import ModelService
from controllers.audio_controller import AudioController

def create_app(config_class: AppConfig = config):
    """Application factory pattern"""

    app = Flask(__name__)

    app.config.update(config_class.settings.flask_config)

    # Initialize CORS
    CORS(app, origins=config_class['CORS_ORIGINS'])

    # Setup logging
    setup_logger(app)

    # Initialize services (dependency injection)
    model_service = ModelService(config_class)

    # Initialize controllers with dependencies
    audio_controller = AudioController(model_service)

    # Register blueprints
    app.register_blueprint(audio_controller.blueprint)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=app.config['DEBUG'],
        port=app.config['PORT'],
        host=app.config['HOST']
    )