# main.py

import os
import sys
import logging
from pathlib import Path
from cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnn_classifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from cnn_classifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnn_classifier.pipeline.stage_04_model_training import ModelTrainingPipeline
from cnn_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from cnn_classifier.pipeline.stage_06_model_prediction import ModelPredictionPipeline

# --- Logging Setup (as defined in src/cnn_classifier/__init__.py) ---
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cnnClassifierLogger")
# --- End Logging Setup ---


# --- Main execution block for the pipeline ---
# This ensures that the code inside this block only runs when the script is executed directly,
# not when it's imported by child processes (e.g., for DataLoader num_workers).
if __name__ == '__main__':
    # It's good practice to include freeze_support() for Windows, though often not strictly
    # necessary if the main logic is correctly guarded. It helps with executable creation.
    # from multiprocessing import freeze_support
    # freeze_support() # Uncomment if you plan to create executables with PyInstaller/similar

    # --- Stage 1: Data Ingestion ---
    STAGE_NAME = "Data Ingestion stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e

    # --- Stage 2: Data Transformation ---
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_pipeline = DataTransformationTrainingPipeline()
        data_transformation_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e

    # --- Stage 3: Prepare Base Model ---
    STAGE_NAME = "Prepare Base Model stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        prepare_base_model_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e

    # --- Stage 4: Model Training ---
    STAGE_NAME = "Model Training stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
   # --- NEW Stage 5: Model Evaluation ---
    STAGE_NAME = "Model Evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
    # --- Stage 6: Model Prediction ---
    STAGE_NAME = "Model Prediction stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_prediction_pipeline = ModelPredictionPipeline()
        # Define a sample image path for testing the prediction pipeline
        # You MUST ensure this path points to an actual image in your dataset
        sample_image_path = Path("artifacts/data_ingestion/archive (3)/Testing/meningioma/Te-me_0018.jpg") # Example path
        if not sample_image_path.exists():
            logger.warning(f"Sample image for prediction not found at {sample_image_path}. Please update 'sample_image_path' in main.py for testing Stage 6.")
            # Optionally, raise an error or skip the prediction stage if the sample image is critical
            # raise FileNotFoundError(f"Sample image not found for prediction at {sample_image_path}")
        else:
            predicted_class, probabilities = model_prediction_pipeline.main(image_path=sample_image_path)
            logger.info(f"Final Prediction for {sample_image_path.name}: Predicted Class = {predicted_class}, Probabilities = {probabilities}")

        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e