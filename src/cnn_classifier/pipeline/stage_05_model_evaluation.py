# src/cnn_classifier/pipeline/stage_05_model_evaluation.py

from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_evaluation import ModelEvaluation
from cnn_classifier import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main method to execute the model evaluation stage.
        """
        try:
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            model_evaluator = ModelEvaluation(config=evaluation_config)
            metrics = model_evaluator.evaluate()
            logger.info(f"Model evaluation completed. Metrics: {metrics}")
        except Exception as e:
            logger.exception(f"Error during Model Evaluation stage: {e}")
            raise e

if __name__ == '__main__':
    STAGE_NAME = "Model Evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
