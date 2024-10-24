# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# License: BSD
# Author: Sasank Chilamkurthy

import pandas as pd
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import time
from tqdm import tqdm  # Import tqdm for progress bar

cudnn.benchmark = True
plt.ion()   # interactive mode

# Initialize Weights & Biases
wandb.init(project="car_classification", config={
    "learning_rate": 0.001,
    "epochs": 25,
    "batch_size": 4
})

# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CustomCarDataset():
    def __init__(self, csv_file, is_pruned=False, path_to_train_folder='/Users/simonhampp/Desktop/WS2425/ADL/adl-gruppe-1/Code/image_classifier/Resnet50/Data/train', transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode
        self.train_folder = path_to_train_folder
        self.labels = sorted(self.data['brand'].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        
        if not is_pruned:
            # Refactor data to have one row per image file
            self.refactor_data()
            # Remove invalid rows during initialization
            self._clean_invalid_rows()
            self.data.to_csv('data_relevant_pruned.csv', index=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_dir = row['dir_path']
        image_file = row['image_file_name']  # Single file name after refactor
        correct_path = self._reformat_path(os.path.join(image_dir, image_file))

        # Check if path exists
        if not os.path.exists(correct_path):
            print(f"File not found: {correct_path}. Removing row {idx}.")
            self.data.drop(idx, inplace=True)  # Remove row if file doesn't exist
            self.data.reset_index(drop=True, inplace=True)
            return None  # Skip returning the item

        # Load the image
        image = Image.open(correct_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get the label for the image
        label = self.label_map[row['brand']]

        return image, label

    def _reformat_path(self, old_path):
        parts = old_path.split('/')
        
        # Extract necessary parts
        brand = parts[1]
        model = parts[2]
        year_range = parts[3]
        file_name = parts[4]

        # Create the new path
        new_path = os.path.join(f"{model}_{year_range}", file_name)

        return os.path.join(self.train_folder, new_path)

    def _clean_invalid_rows(self):
        # Go through all rows and remove those that have invalid paths
        valid_indices = []
        for idx, row in self.data.iterrows():
            image_dir = row['dir_path']
            image_file = row['image_file_name']  # Single file name after refactor
            correct_path = self._reformat_path(os.path.join(image_dir, image_file))

            if os.path.exists(correct_path):
                valid_indices.append(idx)
        
        # Keep only valid rows
        self.data = self.data.loc[valid_indices].reset_index(drop=True)

    def refactor_data(self):
        # Create a new DataFrame to store the split rows
        refactored_rows = []
        
        for _, row in self.data.iterrows():
            image_files = eval(row['image_file_names'])  # Convert string representation of list to actual list
            for image_file in image_files:
                new_row = row.copy()  # Copy the current row to modify
                new_row['image_file_name'] = image_file  # Create a single image file name column
                refactored_rows.append(new_row)

        # Create the refactored DataFrame
        self.data = pd.DataFrame(refactored_rows)
        self.data.reset_index(drop=True, inplace=True)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data with tqdm progress bar
                with tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}", unit='batch') as tbar:
                    for inputs, labels in tbar:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                # TODO: do i need optimizer.zero_grad?
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        # Update progress bar with current loss
                        tbar.set_postfix(loss=loss.item())

                if phase == 'train':
                    scheduler.step()


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                # Log metrics to W&B
                wandb.log({f"{phase} Loss": epoch_loss, f"{phase} Accuracy": epoch_acc})
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        # Log the model as an artifact
        wandb.save(best_model_params_path)
        
    return model

def prepare_datasets_and_dataloaders(csv_file, path_image_folder, data_transforms, csv_pruned = True):
    # Create datasets for train and validation
    train_dataset = CustomCarDataset(csv_file=csv_file, path_to_train_folder=path_image_folder, is_pruned=csv_pruned, transform=data_transforms['train'], mode='train')
    val_dataset = CustomCarDataset(csv_file=csv_file, path_to_train_folder=path_image_folder, is_pruned=csv_pruned, transform=data_transforms['val'], mode='val')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Store dataloaders in a dictionary for easy access
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Dataset sizes
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    return dataloaders, dataset_sizes, train_dataset

# Load CSV data
csv_file = '/Users/simonhampp/Desktop/WS2425/ADL/adl-gruppe-1/Code/image_classifier/Resnet50/Data/data_relevant_pruned_train.csv'  # Change this to your actual CSV file path
path_image_folder = '/Users/simonhampp/Desktop/WS2425/ADL/adl-gruppe-1/Code/image_classifier/Resnet50/Data/train'

# Prepare datasets and dataloaders
dataloaders, dataset_sizes, train_dataset = prepare_datasets_and_dataloaders(csv_file, path_image_folder, data_transforms, csv_pruned=True)

# Define device
if torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   print ("MPS device not found.")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_conv = torchvision.models.resnet50(weights='IMAGENET1K_V1')
# model_conv = torchvision.models.alexnet()
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(train_dataset.labels))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=1)

# Save the trained model
model_save_path = 'trained_model_on_test.pth'
torch.save(model_conv.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')