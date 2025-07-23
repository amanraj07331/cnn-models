# main.py

import os
import sys
import logging
from cnn_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnn_classifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from cnn_classifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnn_classifier.pipeline.stage_04_model_training import ModelTrainingPipeline
from cnn_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
# from cnn_classifier.pipeline.stage_06_model_prediction import ModelPredictionPipeline # REMOVE THIS IMPORT
from pathlib import Path


# --- Logging Setup ---
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
if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()

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

    # --- Stage 5: Model Evaluation ---
    STAGE_NAME = "Model Evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e

    # --- Stage 6: Model Prediction (This stage is now typically run standalone or via the app) ---
    # You can keep this commented out in main.py, as the app.py will handle live predictions.
    # If you want to test the prediction pipeline specifically without the app, you can uncomment
    # this block and ensure sample_image_path points to a valid image.
    # STAGE_NAME = "Model Prediction stage"
    # try:
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_prediction_pipeline = ModelPredictionPipeline()
    #     sample_image_path = Path("artifacts/data_ingestion/archive (3)/Testing/glioma/Te-gl_0000.jpg")
    #     if not sample_image_path.exists():
    #         logger.warning(f"Sample image for prediction not found at {sample_image_path}. Please update 'sample_image_path' in main.py for testing Stage 6.")
    #     else:
    #         predicted_class, probabilities = model_prediction_pipeline.main(image_path=sample_image_path)
    #         logger.info(f"Final Prediction for {sample_image_path.name}: Predicted Class = {predicted_class}, Probabilities = {probabilities}")
    #     logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    # except Exception as e:
    #     logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
    #     raise e
