# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler

from dataloader import prepare_datasets_and_dataloaders
from transfer_learn import train_model

def configure_device():
   device = None
   if torch.backends.mps.is_available():
      device = torch.device("mps")
      print("Using MPS")
   elif torch.cuda.is_available():
      device = torch.device("cuda:0")
      print("Using GPU")
   else:
      device = torch.device("cpu")
      print("Using CPU")
   return device

def initialize_resnet50_for_transfer_learning(num_classes, device):
   model_conv = torchvision.models.resnet50(weights='IMAGENET1K_V1')
   for param in model_conv.parameters():
      param.requires_grad = False

   # Parameters of newly constructed modules have requires_grad=True by default
   num_ftrs = model_conv.fc.in_features
   model_conv.fc = nn.Linear(num_ftrs, num_classes)

   model_conv = model_conv.to(device)
   return model_conv

def setup_training(model_conv):
   criterion = nn.CrossEntropyLoss()
   optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001, weight_decay=0.0001)
   exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
   return criterion, optimizer_conv, exp_lr_scheduler

if __name__ == '__main__':
   device = configure_device()
   load_everything_in_memory = False
   # Setup data, model and training
   dataloaders, dataset_sizes, train_dataset = prepare_datasets_and_dataloaders(in_memory_flag=load_everything_in_memory)
   model_conv = initialize_resnet50_for_transfer_learning(len(train_dataset.labels), device)
   criterion, optimizer_conv, exp_lr_scheduler = setup_training(model_conv)

   # Train the model
   model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=1)

   # Save the trained model
   model_save_path = 'trained_model_on_train.pth'
   torch.save(model_conv.state_dict(), model_save_path)
   print(f'Model saved to {model_save_path}')