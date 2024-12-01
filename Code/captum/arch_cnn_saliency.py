import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from captum.attr import LayerGradCam

device = torch.device("mps" if torch.backends.mps.is_available() else("cuda" if torch.cuda.is_available() else "cpu"))

# Load ImageNet labels
with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

# Load a pre-trained model
model = resnet50(pretrained=True).to(device=device)
model.eval()

# Choose the target layer for Grad-CAM
target_layer = model.layer4[2].conv3

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the input image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

image_path = 'cat_dog.jpg'  # Replace with your image path
input_tensor, original_image = load_image(image_path)

# CHAT-GPT: Ensure input tensor matches the device and dtype of the model
input_tensor = input_tensor.to(device)
input_tensor = input_tensor.to(dtype=next(model.parameters()).dtype)

# Get top predictions
def get_top_predictions(output, top_k=32):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_idxs = torch.topk(probabilities, top_k)
    return [(labels[idx], top_probs[i].item(), idx) for i, idx in enumerate(top_idxs)]

with torch.no_grad():
    output = model(input_tensor)
    top_predictions = get_top_predictions(output)

# Print top predictions
print("Top Predictions:")
for i, (label, prob, idx) in enumerate(top_predictions):
    print(f"{i}. {label} ({prob * 100:.2f}%) - Class Index: {idx}")

# Ask the user to select a target class from the top
selected_index = 0
target_class = top_predictions[selected_index][2]  # Get the class index of the selected prediction
target_label = top_predictions[selected_index][0]
target_prob = top_predictions[selected_index][1]
print("target_label: " + str(target_label))
print("target_prob: " + str(target_prob))
print("target_class: " + str(target_class))

# Initialize LayerGradCam with the model and target layer
grad_cam = LayerGradCam(model, target_layer)

# Compute Grad-CAM attributions for the selected class
attributions = grad_cam.attribute(input_tensor, target=target_class)

# Helper function to visualize the original image with Grad-CAM heatmap
def show_gradcam(original_image, attributions):
    # Convert attributions to numpy
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.maximum(attributions, 0)  # ReLU, only positive values

    # Resize attributions to match the input image size
    attributions = np.uint8(255 * attributions / attributions.max())
    attributions = Image.fromarray(attributions).resize(original_image.size, Image.BILINEAR)

    # Plot the original image and Grad-CAM heatmap overlay
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(original_image)
    ax[1].imshow(attributions, cmap='jet', alpha=0.5)  # Overlay heatmap with transparency
    ax[1].axis('off')
    ax[1].set_title('Grad-CAM')

    plt.tight_layout()
    plt.show()

# Visualize the Grad-CAM heatmap
show_gradcam(original_image, attributions[0])
