# src/cnn_classifier/pipeline/stage_06_model_prediction.py

from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_prediction import Prediction
from cnn_classifier import logger
from pathlib import Path # Import Path for sample image path

class ModelPredictionPipeline:
    def __init__(self):
        pass

    def main(self, image_path: Path):
        """
        Main method to execute the model prediction stage.
        Args:
            image_path (Path): Path to the image file to predict on.
        """
        try:
            config = ConfigurationManager()
            prediction_config = config.get_prediction_config()
            predictor = Prediction(config=prediction_config)

            # Make a prediction
            predicted_class, probabilities = predictor.predict(image_path)
            logger.info(f"Prediction for image '{image_path.name}': {predicted_class} with probabilities: {probabilities}")
            return predicted_class, probabilities
        except Exception as e:
            logger.exception(f"Error during Model Prediction stage for image {image_path}: {e}")
            raise e

if __name__ == '__main__':
    STAGE_NAME = "Model Prediction stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        # IMPORTANT: Provide a path to a sample image for testing here.
        # This image should be within your project structure or a known path.
        # For example, if you have a test image in artifacts/data_ingestion/archive (3)/Testing/glioma/Te-gl_0000.jpg
        sample_image_path = Path("artifacts/data_ingestion/archive (3)/Testing/meningioma/Te-me_0018.jpg")
        # Ensure this path exists and points to an actual image for testing.
        if not sample_image_path.exists():
            logger.warning(f"Sample image not found at {sample_image_path}. Please update 'sample_image_path' in stage_06_model_prediction.py for testing.")
        else:
            predicted_class, probabilities = obj.main(image_path=sample_image_path)
            logger.info(f"Test prediction for {sample_image_path.name}: Predicted Class = {predicted_class}, Probabilities = {probabilities}")

        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
