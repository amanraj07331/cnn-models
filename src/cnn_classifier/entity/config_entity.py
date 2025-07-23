# src/cnn_classifier/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    image_size: list
    batch_size: int
    is_augmentation: bool
    num_classes: int
    training_data: Path
    testing_data: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_batch_size: int
    params_is_augmentation: bool
    params_num_classes: int
    params_learning_rate: float
    params_epochs: int
    params_patience: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    testing_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_num_classes: int
    params_patience: int

@dataclass(frozen=True)
class EvaluationConfig: # NEW DATACLASS FOR EVALUATION
    root_dir: Path
    model_path: Path
    testing_data: Path
    image_size: list
    batch_size: int
    num_classes: int
    class_names: list # To map predicted indices back to class names

@dataclass(frozen=True)
class PredictionConfig: # Renamed from previous Stage 5, now Stage 6
    root_dir: Path
    model_path: Path
    image_size: list
    num_classes: int
    class_names: list
    
@dataclass(frozen=True)
class DeploymentConfig: # NEW DATACLASS FOR DEPLOYMENT
    root_dir: Path
    model_path: Path
    host: str
    port: int
    upload_folder: Path
    image_size: list # Needed for prediction within the app
    num_classes: int # Needed for prediction within the app
    class_names: list # Needed for prediction within the app    