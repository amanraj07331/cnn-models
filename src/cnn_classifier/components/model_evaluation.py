# src/cnn_classifier/components/model_evaluation.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import EvaluationConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json # To save evaluation metrics

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for model evaluation: {self.device}")

        self.model = self._load_model()
        self.test_loader = self._get_test_loader()
        self.class_names = self.config.class_names
        logger.info(f"Evaluation component initialized. Model loaded, test loader ready.")

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
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.to(self.device) # Move model to the appropriate device
        model.eval() # Set model to evaluation mode (important for inference)
        logger.info(f"Model loaded successfully from {self.config.model_path}.")
        return model

    def _get_test_loader(self):
        """
        Prepares the DataLoader for the testing dataset.
        This should use the same transformations as the validation/test set during training.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            normalize,
        ])

        test_dataset = datasets.ImageFolder(
            root=self.config.testing_data,
            transform=test_transform
        )
        cpu_count=os.cpu_count() or 0
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=cpu_count
        )
        logger.info(f"Test DataLoader created with {len(test_loader)} batches from {self.config.testing_data}.")
        return test_loader

    def evaluate(self):
        """
        Performs a comprehensive evaluation of the model on the test set
        and saves the metrics.
        """
        self.model.eval() # Set model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient calculation for evaluation
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds).tolist() # Convert to list for JSON serialization

        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }

        # Create evaluation root directory if it doesn't exist
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        # Define path to save metrics
        metrics_path = self.config.root_dir / "evaluation_metrics.json"

        # Save metrics to a JSON file
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation metrics saved to: {metrics_path}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{json.dumps(report, indent=4)}")
        logger.info(f"Confusion Matrix:\n{json.dumps(cm, indent=4)}")

        return metrics

