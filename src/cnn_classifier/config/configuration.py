# src/cnn_classifier/config/configuration.py

from cnn_classifier.constants import *
from cnn_classifier.utils.common import read_yaml, create_directories
from cnn_classifier.entity.config_entity import (DataIngestionConfig,
                                                  DataTransformationConfig,
                                                  PrepareBaseModelConfig,
                                                  TrainingConfig,
                                                  EvaluationConfig, # Import new dataclass
                                                  PredictionConfig)
from box.exceptions import BoxValueError

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            image_size=list(params.IMAGE_SIZE),
            batch_size=params.BATCH_SIZE,
            is_augmentation=params.IS_AUGMENTATION,
            num_classes=params.NUM_CLASSES,
            training_data=Path(config.training_data),
            testing_data=Path(config.testing_data)
        )
        return data_transformation_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=list(params.IMAGE_SIZE),
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.IS_AUGMENTATION,
            params_num_classes=params.NUM_CLASSES,
            params_learning_rate=params.LEARNING_RATE,
            params_epochs=params.EPOCHS,
            params_patience=params.PATIENCE
        )
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.model_training
        prepare_base_model_config = self.config.prepare_base_model
        data_transformation_config = self.config.data_transformation
        params = self.params

        create_directories([training_config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            training_data=Path(data_transformation_config.training_data),
            testing_data=Path(data_transformation_config.testing_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.IS_AUGMENTATION,
            params_image_size=list(params.IMAGE_SIZE),
            params_learning_rate=params.LEARNING_RATE,
            params_num_classes=params.NUM_CLASSES,
            params_patience=params.PATIENCE
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig: # NEW METHOD FOR EVALUATION
        """
        Retrieves model evaluation configuration from config.yaml and params.yaml.
        Creates the root directory for evaluation artifacts.
        Returns:
            EvaluationConfig: Configuration object for model evaluation.
        """
        eval_config = self.config.model_evaluation
        params = self.params
        data_transformation_config = self.config.data_transformation # To get testing_data path

        create_directories([eval_config.root_dir])

        # Define class names based on your dataset structure
        # (alphabetical order of folder names).
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

        evaluation_config = EvaluationConfig(
            root_dir=Path(eval_config.root_dir),
            model_path=Path(eval_config.model_path),
            testing_data=Path(data_transformation_config.testing_data), # Use path from data_transformation
            image_size=list(params.IMAGE_SIZE),
            batch_size=params.BATCH_SIZE,
            num_classes=params.NUM_CLASSES,
            class_names=class_names
        )
        return evaluation_config

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.model_prediction
        params = self.params

        create_directories([config.root_dir])

        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

        prediction_config = PredictionConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            image_size=list(params.IMAGE_SIZE),
            num_classes=params.NUM_CLASSES,
            class_names=class_names
        )
        return prediction_config
    
