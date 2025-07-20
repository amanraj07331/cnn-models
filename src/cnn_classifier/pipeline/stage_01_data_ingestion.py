from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion import DataIngestion
from cnn_classifier import logger


STAGE_NAME ="Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init_(self):
        pass
    def main(self):
     config = ConfigurationManager()
     data_ingestion_config = config.get_data_ingestion_config()

     data_ingestion = DataIngestion(config=data_ingestion_config)
     data_ingestion.download_file()
     data_ingestion.extract_zip_file()

     logger.info("🎉 Data ingestion completed successfully!")
     
     
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
     logger.exception(f"❌ Data ingestion failed due to: {e}")
     raise e