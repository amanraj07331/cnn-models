# src/cnn_classifier/components/model_prediction.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image # For image loading
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import PredictionConfig

class Prediction:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for prediction: {self.device}")

        self.model = self._load_model()
        self.transform = self._get_transform()
        self.class_names = self.config.class_names
        logger.info(f"Prediction component initialized. Model loaded, transforms ready.")

    def _load_model(self):
        """
        Loads the trained model from the specified path.
        """
        # Initialize ResNet18 architecture (without pre-trained weights initially)
        model = models.resnet18(weights=None)
        # Adjust the final fully connected layer to match your number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.config.num_classes)

        # Load the saved state dictionary (trained weights)
        # map_location ensures it loads correctly regardless of original training device
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device) # Move model to the appropriate device
        model.eval() # Set model to evaluation mode (important for inference)
        logger.info(f"Model loaded successfully from {self.config.model_path}.")
        return model

    def _get_transform(self):
        """
        Defines the image transformation pipeline for prediction.
        This should match the validation/test transforms used during training.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size), # Resize to the expected input size
            transforms.ToTensor(), # Convert to PyTorch Tensor
            normalize, # Normalize pixel values
        ])
        logger.info("Image transformation pipeline for prediction created.")
        return transform

    def predict(self, image_path: Path) -> tuple[str, list]:
        """
        Makes a prediction on a single input image.

        Args:
            image_path (Path): Path to the input image file.

        Returns:
            tuple[str, list]: The predicted class name and list of probabilities.
        """
        if not image_path.exists():
            logger.error(f"Image not found at: {image_path}")
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB") # Open image and ensure RGB format
            logger.info(f"Image loaded: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise ValueError(f"Could not load image: {image_path}")

        # Apply transformations
        input_tensor = self.transform(image)
        logger.info(f"Type of input_tensor after transform: {type(input_tensor)}")
        if not isinstance(input_tensor, torch.Tensor):
            logger.error("Transformation failed: input_tensor is not a torch.Tensor!")
            raise TypeError("Expected input_tensor to be a torch.Tensor after transformation.")
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        logger.info(f"Image transformed and moved to device. Shape: {input_batch.shape}")

        with torch.no_grad(): # Disable gradient calculations for inference
            output = self.model(input_batch)
            probabilities = torch.softmax(output, dim=1) # Get probabilities
            _, predicted_idx = torch.max(probabilities, 1) # Get the index of the highest probability
            predicted_class_idx = int(predicted_idx.item()) # Get the scalar index
            predicted_class_name = self.class_names[predicted_class_idx] # Map index to class name

        logger.info(f"Prediction for {image_path}: Class Index {predicted_class_idx}, Class Name: {predicted_class_name}")
        return predicted_class_name, probabilities.cpu().numpy().tolist()[0] # Return name and probabilities for more info
