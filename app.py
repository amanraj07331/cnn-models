# app.py

import os
import sys
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from pathlib import Path 
import logging
from typing import Optional # Import Optional

# Add the src directory to the Python path so we can import cnn_classifier
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / "src"))

# Import necessary modules from your project
from cnn_classifier.utils.common import create_directories
from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_prediction import Prediction
from cnn_classifier.entity.config_entity import PredictionConfig, DeploymentConfig 

# --- Logging Setup ---
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs_app.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cnnAppLogger")
# --- End Logging Setup ---

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and config (initialized once)
# --- FIX STARTS HERE ---
predictor: Optional[Prediction] = None # Use Optional for predictor
# --- FIX ENDS HERE ---
deployment_config: Optional[DeploymentConfig] = None 

def initialize_predictor():
    """
    Initializes the Prediction component and loads the model.
    This function is called once when the Flask app starts.
    """
    global predictor, deployment_config
    try:
        config_manager = ConfigurationManager()
        deployment_config = config_manager.get_deployment_config() 
        create_directories([deployment_config.upload_folder])

        prediction_component_config = PredictionConfig(
            root_dir=deployment_config.root_dir,
            model_path=deployment_config.model_path,
            image_size=deployment_config.image_size,
            num_classes=deployment_config.num_classes,
            class_names=deployment_config.class_names
        )

        predictor = Prediction(config=prediction_component_config)
        logger.info("Prediction service initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize prediction service: {e}")
        sys.exit(1)

# Call initialization function when app starts
with app.app_context():
    initialize_predictor()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    Renders the main index.html page for image upload.
    """
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    """
    API endpoint to receive an image, make a prediction, and return the result.
    """
    assert deployment_config is not None, "Deployment configuration not initialized."
    # --- FIX STARTS HERE ---
    assert predictor is not None, "Prediction service not initialized." # Assert predictor is not None
    # --- FIX ENDS HERE ---

    if 'image' not in request.files:
        logger.warning("No image file part in the request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    if not file or file.filename == '':
        logger.warning("No selected file or empty filename.")
        return jsonify({"error": "No selected image file"}), 400

    filename = file.filename
    filepath = deployment_config.upload_folder / str(filename) 

    try:
        file.save(filepath) 
        logger.info(f"Uploaded image saved to: {filepath}")

        predicted_class, probabilities = predictor.predict(image_path=filepath) 
        logger.info(f"Prediction made: {predicted_class}")

        os.remove(filepath)
        logger.info(f"Deleted uploaded image: {filepath}")

        return jsonify({
            "prediction": predicted_class,
            "probabilities": probabilities,
            "message": "Prediction successful"
        })
    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


if __name__ == "__main__":
    assert deployment_config is not None, "Deployment configuration not initialized for app.run."
    # --- FIX STARTS HERE ---
    assert predictor is not None, "Prediction service not initialized for app.run." # Assert predictor here too
    # --- FIX ENDS HERE ---
    logger.info(f"Starting Flask application on {deployment_config.host}:{deployment_config.port}")
    app.run(host=deployment_config.host, port=deployment_config.port, debug=True)
