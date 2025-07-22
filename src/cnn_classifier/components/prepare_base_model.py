# src/cnn_classifier/components/prepare_base_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        # Determine the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for model preparation: {self.device}")

    def get_base_model(self):
        """
        Loads a pre-trained ResNet18 model and freezes its parameters.
        Saves the initial frozen base model.
        """
        # Load pre-trained ResNet18 model with best available weights
        # models.ResNet18_Weights.IMAGENET1K_V1 represents the best available weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        logger.info("Loaded pre-trained ResNet18 model.")

        # Freeze all parameters in the feature extracting layers
        # This prevents their weights from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Frozen all parameters in the base model (feature extractor).")

        # Create the root directory for saving models if it doesn't exist
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        # Save the state dictionary of the initial frozen base model
        # This saves only the model's learned parameters, not the architecture
        torch.save(self.model.state_dict(), self.config.base_model_path)
        logger.info(f"Initial base model (state_dict) saved to: {self.config.base_model_path}")

    def update_base_model(self):
        """
        Modifies the final classification layer of the base model
        to match the number of output classes for our specific problem.
        Moves the model to the appropriate device (CPU/GPU) and saves it.
        """
        # Ensure the base model is loaded and frozen before updating
        if not hasattr(self, 'model'):
            self.get_base_model() # Call get_base_model to load and freeze if not already done

        # Get the number of input features for the final fully connected layer
        num_ftrs = self.model.fc.in_features

        # Replace the final fully connected layer with a new one
        # The new layer will have 'num_ftrs' input features and 'params_num_classes' output features
        self.model.fc = nn.Linear(num_ftrs, self.config.params_num_classes)
        logger.info(f"Modified final FC layer to have {self.config.params_num_classes} output features.")

        # Move the entire model (including the new FC layer) to the specified device
        self.model.to(self.device)
        logger.info(f"Model moved to {self.device}.")

        # Save the state dictionary of the updated base model
        # This model now has the modified classifier head
        torch.save(self.model.state_dict(), self.config.updated_base_model_path)
        logger.info(f"Updated base model (state_dict) saved to: {self.config.updated_base_model_path}")

