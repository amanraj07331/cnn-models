import os
import urllib.request as request
import zipfile 
from cnn_classifier import logger
from cnn_classifier.utils.common import get_size
from cnn_classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import gdown

# Data ingestion logic
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading from Google Drive...")
            gdown.download(
                url=self.config.source_URL,
                output=str(self.config.local_data_file),
                quiet=False
            )
            logger.info(f"✅ File downloaded to: {self.config.local_data_file}")
        else:
            logger.info(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"✅ Extracted zip to: {unzip_path}")