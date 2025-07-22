# src/cnn_classifier/components/model_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets, models
from pathlib import Path
from cnn_classifier import logger
from cnn_classifier.entity.config_entity import TrainingConfig
from cnn_classifier.utils.common import save_bin # Assuming save_bin is in common.py
import time
import copy

class ModelTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for model training: {self.device}")

        # Load the updated base model from the previous stage
        self.model = models.resnet18(weights=None) # Initialize ResNet18 without pre-trained weights first
        # Modify the final layer to match the number of classes (as done in prepare_base_model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config.params_num_classes)
        # Load the state_dict from the updated base model path
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Loaded updated base model from {self.config.updated_base_model_path} to {self.device}.")

        # Define loss function, optimizer, and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        # Learning rate scheduler: decreases LR by a factor of 0.1 every 7 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        logger.info("Criterion, Optimizer, and Scheduler initialized.")

    def get_data_loaders(self):
        """
        Re-initializes DataLoaders for training and validation within the training component.
        This is necessary because the DataTransformation component returns the loaders,
        but for modularity, the Training component should be able to get its own loaders
        based on the config paths.
        """
        # Standard ImageNet normalization values
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Training data augmentation and transformation
        train_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size),
            transforms.RandomHorizontalFlip() if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize,
        ])

        # Validation/Testing data transformation
        val_test_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size),
            transforms.ToTensor(),
            normalize,
        ])

        full_train_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root=self.config.testing_data,
            transform=val_test_transform
        )

        # Split the full training dataset into actual training and validation sets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
        cpu_count = os.cpu_count() or 0
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=cpu_count,
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=cpu_count,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=cpu_count,
        )
        logger.info("DataLoaders prepared for training.")


    def train_model(self):
        """
        Executes the training loop for the model.
        Includes validation, learning rate scheduling, and early stopping.
        Saves the best model based on validation accuracy.
        """
        self.get_data_loaders() # Ensure data loaders are ready

        since = time.time() # Start time for training duration

        best_model_wts = copy.deepcopy(self.model.state_dict()) # Store best model weights
        best_acc = 0.0 # Track best validation accuracy
        epochs_no_improve = 0 # Counter for early stopping

        for epoch in range(self.config.params_epochs):
            logger.info(f'Epoch {epoch+1}/{self.config.params_epochs}')
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = self.train_loader
                else:
                    self.model.eval()   # Set model to evaluate mode
                    dataloader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    # Track history only in train phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset) # type: ignore
                epoch_acc = running_corrects.double() / len(dataloader.dataset) # type: ignore

                logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model if it's the best validation accuracy found so far
                if phase == 'val':
                    self.scheduler.step() # Step the learning rate scheduler after validation
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        epochs_no_improve = 0 # Reset early stopping counter
                        logger.info(f"New best validation accuracy: {best_acc:.4f}. Saving model weights.")
                        # Save the best model state dictionary
                        torch.save(self.model.state_dict(), self.config.trained_model_path)
                    else:
                        epochs_no_improve += 1
                        logger.info(f"Validation accuracy did not improve. No improvement for {epochs_no_improve} epochs.")
                        if epochs_no_improve >= self.config.params_patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {self.config.params_patience} epochs.")
                            break # Break out of phase loop

            if epochs_no_improve >= self.config.params_patience:
                break # Break out of epoch loop if early stopping triggered

        time_elapsed = time.time() - since
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        # The best model is already saved during training, but this ensures the current
        # self.model object holds the best weights.
        logger.info(f"Final trained model (best weights) saved to: {self.config.trained_model_path}")

    def evaluate_model(self):
        """
        Evaluates the trained model on the test dataset.
        """
        self.model.eval() # Set model to evaluation mode
        running_corrects = 0
        total_samples = 0

        with torch.no_grad(): # Disable gradient calculation for evaluation
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        accuracy = running_corrects.double() / total_samples # type: ignore
        logger.info(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

