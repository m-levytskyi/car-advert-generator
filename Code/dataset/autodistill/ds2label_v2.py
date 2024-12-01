import os
import concurrent.futures
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import torch
from PIL import Image
import cv2

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

ontology = CaptionOntology({
    "sports_car": "sports_car",
    "family_car": "family_car",
    "hatchback": "hatchback",
    "sedan": "sedan",
    "suv": "suv",
    "alfaromeo": "ALFAROMEO",
    "chevrolet": "CHEVROLET",
    "peugeot": "PEUGEOT",
    "citroen": "CITROEN",
    "hyundai": "HYUNDAI",
    "renault": "RENAULT",
    "nissan": "NISSAN",
    "mazda": "MAZDA",
    "honda": "HONDA",
    "lexus": "LEXUS",
    "volvo": "VOLVO",
    "skoda": "SKODA",
    "fiat": "FIAT",
    "opel": "OPEL",
    "ford": "FORD",
    "kia": "KIA",
})

# OUTPUT - LABELED DATASET
IMAGE_DIR_PATH = f'../Data/DS2_b2000/'

# INPUT - UNLABELED DATASET
DATASET_DIR_PATH = f'../Data/DS2_b2000_labels/'

# Iterate over the subdirectories
for root, dirs, files in os.walk(IMAGE_DIR_PATH):
    for dir_name in dirs:
        print(f"Subdirectory: {os.path.join(root, dir_name)}")

        input = os.path.join(root, dir_name)
        output = os.path.join(DATASET_DIR_PATH, dir_name)

        # Initialize the base model
        base_model = GroundingDINO(ontology=ontology)

        # Check if GroundingDINO has internal models that can be moved to the device
        if hasattr(base_model, 'model'):
            base_model.model = base_model.model.to(device)

        # Use the model to label the dataset
        dataset = base_model.label(
            input_folder=input,
            output_folder=output,
            extension=".jpg"
        )

