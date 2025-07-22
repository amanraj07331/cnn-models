# src/cnn_classifier/pipeline/stage_03_prepare_base_model.py

from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.prepare_base_model import PrepareBaseModel
from cnn_classifier import logger

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model() # This loads and saves the initial frozen model
        prepare_base_model.update_base_model() # This updates the FC layer and saves the final model

if __name__ == '__main__':
    STAGE_NAME = "Prepare Base Model stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
