import os
import json
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from tqdm import tqdm

# Path to the image folder
image_folder = '/Users/johannesdecker/ADL_DATASETS/DS2'

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Load the ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.to(device)
model.eval()

brands = [
    "volkswagen", "toyota", "renault", "nissan", "hyundai", 
    "ford", "honda", "chevrolet", "kia", "mercedesbenz", 
    "bmw", "peugeot", "audi", "suzuki", "skoda", 
    "fiat", "mazda", "citroen", "subaru", "opel", 
    "jeep", "mitsubishi", "volvo", "landrover", "seat", 
    "dacia", "lexus", "mini", "porsche", "jaguar", 
    "alfaromeo", "tesla", "buick", "chrysler", "dodge", 
    "ram", "infiniti", "lincoln", "acura", "cadillac", 
    "genesis", "lancia", "mahindra", "ssangyong", "proton", 
    "geely", "greatwall", "mg", "byd", "chery"
]

body_styles = [
    "sports_car", "family_car", "hatchback", "sedan", "suv"
]

classes = body_styles

# Create a dictionary to map the model's output indices to the classes
class_dict = {i: label for i, label in enumerate(classes)}

# Dictionary to store image labels
labels = {}

# Get the list of images
image_list = os.listdir(image_folder)

# Process each image with a progress bar
for image_name in tqdm(image_list, desc="Labeling images"):
    image_path = os.path.join(image_folder, image_name)
    try:
        # Preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        continue

    with torch.no_grad():
        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        best_match = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, best_match].item()

        # Save the best match and its confidence
        labels[image_name] = {
            "label": class_dict.get(best_match, "unknown"),
            "confidence": confidence
        }

# Save labels to a JSON file in the image folder
labels_path = os.path.join(image_folder, "image_labels_vit-base-patch16-224.json")
with open(labels_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"Labeling complete. Labels saved to {labels_path}.")
