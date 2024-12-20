import os
# from autodistill_clip import CLIP # => causes library / package problems
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import torch
import platform

device = "cuda" if torch.cuda.is_available() else "cpu"

ontology=CaptionOntology({
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
from torch.amp import autocast

print("Python version: " + str(platform.python_version()))

for root, dirs, files in os.walk(IMAGE_DIR_PATH):
      for dir_name in dirs:
        print(f"Subdirectory: {os.path.join(root, dir_name)}")

        input = os.path.join(root, dir_name)
        output = os.path.join(DATASET_DIR_PATH, dir_name)

        with autocast("cuda"):
          base_model = GroundingDINO(ontology=ontology)

          if hasattr(base_model, 'model'):
            base_model.model = base_model.model.to(device)

          dataset = base_model.label(
            input_folder=input,
            output_folder=output,
          extension=".jpg")
