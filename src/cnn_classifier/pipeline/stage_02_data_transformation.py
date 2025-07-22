# src/cnn_classifier/pipeline/stage_02_data_transformation.py

from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_transformation import DataTransformation
from cnn_classifier import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Main method to execute the data transformation stage.
        """
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            # Call the method to get data loaders.
            # The loaders are returned but not explicitly saved here; they will be passed
            # to the subsequent training stage.
            train_loader, val_loader, test_loader, class_names = data_transformation.get_data_loaders()
            logger.info(f"DataLoaders successfully created and classes identified: {class_names}")
        except Exception as e:
            logger.exception(f"Error during Data Transformation stage: {e}")
            raise e

if __name__ == '__main__':
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"❌ Stage {STAGE_NAME} failed: {e}")
        raise e
