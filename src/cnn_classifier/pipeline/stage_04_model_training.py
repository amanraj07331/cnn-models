# src/cnn_classifier/pipeline/stage_04_model_training.py

from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_training import ModelTraining
from cnn_classifier import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main method to execute the model training stage.
        """
        try:
            config = ConfigurationManager()
            training_config = config.get_training_config()
            model_trainer = ModelTraining(config=training_config)
            model_trainer.train_model() # This will handle loading data, training, and saving the best model
            model_trainer.evaluate_model() # Evaluate the best model on the test set
            logger.info("Model training and evaluation completed successfully.")
        except Exception as e:
            logger.exception(f"Error during Model Training stage: {e}")
            raise e

if __name__ == '__main__':
    STAGE_NAME = "Model Training stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
