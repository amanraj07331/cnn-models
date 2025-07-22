# src/cnn_classifier/components/data_transformation.py

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split # Import random_split for train/val split
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        logger.info(f"Data Transformation initialized with config: {self.config}")

    def get_data_loaders(self):
        """
        Applies transformations and creates PyTorch DataLoaders for training,
        validation, and testing datasets.
        Splits the training data into training and validation subsets.
        Returns:
            tuple: (train_loader, val_loader, test_loader, class_names)
        """
        # Standard ImageNet normalization values
        # These are commonly used for models pre-trained on ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Define transformations for the training dataset
        # Includes data augmentation if self.config.is_augmentation is True
        train_transform = transforms.Compose([
            transforms.Resize(self.config.image_size), # Resize images to the specified size
            # Apply random horizontal flip only if augmentation is enabled
            transforms.RandomHorizontalFlip() if self.config.is_augmentation else transforms.Lambda(lambda x: x),
            # Apply random rotation only if augmentation is enabled
            transforms.RandomRotation(10) if self.config.is_augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to PyTorch Tensor
            normalize, # Normalize tensor with mean and standard deviation
        ])

        # Define transformations for validation and testing datasets
        # Typically, no augmentation is applied to validation/test sets
        val_test_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            normalize,
        ])

        # Load the full training dataset using ImageFolder
        # ImageFolder expects data organized in subdirectories by class (e.g., train/glioma, train/meningioma)
        full_train_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=train_transform
        )
        logger.info(f"Loaded full training dataset from: {self.config.training_data}")

        # Load the testing dataset
        test_dataset = datasets.ImageFolder(
            root=self.config.testing_data,
            transform=val_test_transform
        )
        logger.info(f"Loaded testing dataset from: {self.config.testing_data}")

        # Determine class names from the training dataset
        class_names = full_train_dataset.classes
        logger.info(f"Detected classes: {class_names}")

        # Split the full training dataset into actual training and validation sets
        # This is crucial for monitoring model performance during training on unseen data
        train_size = int(0.8 * len(full_train_dataset)) # 80% for training
        val_size = len(full_train_dataset) - train_size # Remaining 20% for validation

        # Use random_split to create reproducible splits (optional: add a generator for fixed random state)
        train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
        logger.info(f"Split training data: {len(train_subset)} samples for training, {len(val_subset)} samples for validation.")

        # Create DataLoaders for each subset
        # DataLoader wraps an iterable around the Dataset to enable easy access to batches
        cpu_count = os.cpu_count() or 0
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True, # Shuffle training data for better generalization
            num_workers=cpu_count,
        ) # <--- This closing parenthesis should be on its own line after num_workers, or before the comment
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=cpu_count,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=cpu_count,
        )

        logger.info(f"Train DataLoader created with {len(train_loader)} batches.")
        logger.info(f"Validation DataLoader created with {len(val_loader)} batches.")
        logger.info(f"Test DataLoader created with {len(test_loader)} batches.")

        return train_loader, val_loader, test_loader, class_names