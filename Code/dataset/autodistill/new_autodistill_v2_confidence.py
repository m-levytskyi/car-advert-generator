import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

image_folder = f'/Users/johannesdecker/ADL_DATASETS/DS2'

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Load the CLIP model and preprocessing function
model, preprocess = clip.load("ViT-L/14", device=device) # "ViT-B/32"

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

# Create text inputs for each class
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

# Compute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Dictionary to store image labels
labels = {}

# Get the list of images
image_list = os.listdir(image_folder)

# Process each image with a progress bar
for image_name in tqdm(image_list, desc="Labeling images"):
    image_path = os.path.join(image_folder, image_name)
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        continue

    with torch.no_grad():
        # Compute image features
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity between image and text features
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_match = similarities.argmax(dim=-1).item()
        confidence = similarities[0, best_match].item()

        # Save the best match and its confidence
        labels[image_name] = {
            "label": classes[best_match],
            "confidence": confidence
        }

# Save labels to a JSON file in the image folder
labels_path = os.path.join(image_folder, "image_labels.json")
with open(labels_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"Labeling complete. Labels saved to {labels_path}.")
