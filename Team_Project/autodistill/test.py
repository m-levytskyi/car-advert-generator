from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2
import os
import torch

# Check if MPS (Metal Performance Shaders) is available for PyTorch
# device = torch.device('mps') if torch.backends.mps.is_available() else(torch.device('cuda') if torch.backends.cpu.is_available() else torch.device('cpu'))
torch.set_default_device('mps')

HOME = os.getcwd()
print("current working directory: " + str(HOME))

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations

base_model = GroundedSAM(ontology=CaptionOntology({"brand": "brand",
                                                   "model": "model",
                                                   "vehicle class": "body_style",
                                                   "segment": "segment",
                                                   "engine": "engine_specs_title",
                                                   "cylinders": "cylinders",
                                                   "displacement": "displacement",
                                                   "torque": "torque",
                                                   "fuel": "fuel",
                                                   "drive type": "drive_type",
                                                   "gearbox": "gearbox",
                                                   "weight": "unladen_weight",
                                                   "wheelbase": "wheelbase",
                                                   "cargo volume": "cargo_volume",
                                                   "top speed": "top_speed",
                                                   "market launch": "from_year"}))

# DATASET_DIR_PATH = f"{HOME}/dataset"
# IMAGE_DIR_PATH = f"{HOME}/images/"



