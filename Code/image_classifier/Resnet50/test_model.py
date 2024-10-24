# Note: Untested code!

import torch
from torch import nn
from torchvision import models
from dataloader import get_test_dataloader

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

# Instantiate the ResNet50 model and modify the fully connected layer
def get_transfer_learned_model(num_classes):
    # Load the pretrained ResNet50 model
    model = models.resnet50(pretrained=False)
    
    # Replace the final fully connected layer with a custom one for the number of classes in your problem
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust 'num_classes' based on your dataset
    
    return model

# Load the number of classes from your test dataset (assuming it's available via the dataloader)
test_loader, dataset_size, test_dataset = get_test_dataloader()
num_classes = len(test_dataset.labels)  # Get the number of output classes

# Create the model with the modified fully connected layer
model = get_transfer_learned_model(num_classes)

# Load the model's state dictionary
model.load_state_dict(torch.load('Code/image_classifier/Resnet50/trained_model_on_test.pth', map_location='cpu'))

# Move the model to the appropriate device
device = configure_device()
model = model.to(device)
model.eval()

# Evaluation loop
running_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

# Calculate and print test accuracy
test_acc = running_corrects.double() / dataset_size
print(f'Test Accuracy: {test_acc:.4f}')
