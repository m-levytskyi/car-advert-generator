import os
import concurrent.futures
# from autodistill_clip import CLIP # => causes library / package problems
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import torch
import clip

import torch
import clip
from PIL import Image
import os
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

ontology=CaptionOntology({
      "roadster_convertible": "roadster_convertible",
      "compact": "compact",
      "compact_suv": "compact_suv",
      "coupe": "coupe",
      "coupe_cabrio": "coupe_cabrio",
      "crossover": "crossover",
      "entry_premium": "entry_premium",
      "exotic": "exotic",
      "large": "large",
      "large_mpv": "large_mpv",
      "large_suv": "large_suv",
      "lower_premium": "lower_premium",
      "luxury": "luxury",
      "medium": "medium",
      "medium_mpv": "medium_mpv",
      "medium_premium": "medium_premium",
      "medium_suv": "medium_suv",
      "midsize_pickup": "midsize_pickup",
      "mini": "mini",
      "pickup": "pickup",
      "premium_coupe": "premium_coupe",
      "premium_suv": "premium_suv",
      "small": "small",
      "small_mpv": "small_mpv",
      "small_suv": "small_suv",
      "upper_premium": "upper_premium",
      "convertible": "convertible",
      "coupe": "coupe",
      "hatchback": "hatchback",
      "sedan": "sedan",
      "suv": "suv",
      "truck": "truck",
      "van": "van",
      "wagon": "wagon",
      "mercedesbenz": "mercedesbenz",
      "alfaromeo": "alfaromeo",
      "audi": "audi",
      "bmw": "bmw",
      "chevrolet": "chevrolet",
      "citroen": "citroen",
      "ferrari": "ferrari",
      "fiat": "fiat",
      "ford": "ford",
      "honda": "honda",
      "hyundai": "hyundai",
      "kia": "kia",
      "lexus": "lexus",
      "mazda": "mazda",
      "nissan": "nissan",
      "opel": "opel",
      "peugeot": "peugeot",
      "porsche": "porsche",
      "renault": "renault",
      "skoda": "skoda",
      "toyota": "toyota",
      "volkswagen": "volkswagen",
      "volvo": "volvo",
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

      base_model = GroundingDINO(ontology=ontology)

      dataset = base_model.label(
        input_folder=input,
        output_folder=output,
        extension=".jpg")
