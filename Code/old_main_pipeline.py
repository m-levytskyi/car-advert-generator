print("**** Loading. Please wait ****")

from webcam.webcam_capture import WebcamCapture
from image_classifier.classify import CarClassifier
from article_agent.agent_pipeline import AgentPipeline
from article_assembler.assembler_pipeline import AssemblerPipeline

import os
import pandas as pd
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path
import shutil

webcam_imgs_path = "Code/webcam/webcam_images"

# csv has 5 body styles and 22 brands
csv = "Code/dataset/data/reduced_dataset_adjusted.csv"

json_path = "Code/article_agent/json/output.json" #article agent output
images_path = "Code/article_assembler/tmp/imgs" #generated images will be stored here
output_pdf_path = "Code/article.pdf"
 
model_name = "alexnet" #"vit", "alexnet" or "resnet"

# 22 Classes - brands
# FIXME weights now have 23 classes, forced to add Ferrari to the list
weights_brand = "Code/image_classifier/alexnet/alexnet_epoch89_bestTrainLoss_bestValAccuracy.pth"
# 5 Classes - body styles
weights_body = "Code/image_classifier/alexnet/alexnet_body-style_epoch80_loss0.04466895014047623_weights.pth"

def preprocess_images(input_path, output_path):
    """Detect cars, crop to largest bounding box, resize to 256x256"""
    print("Preprocessing images...")
    
    # Initialize YOLO model
    model = YOLO("yolo11x.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get all images in input directory
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [str(path) for path in Path(input_path).glob("*") 
                  if path.suffix.lower() in valid_extensions]
    
    for img_path in image_paths:
        # Load and run YOLO detection
        results = model(img_path, verbose=False)
        
        if len(results[0].boxes) > 0:
            # Get largest bounding box (by area)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_box = boxes[np.argmax(areas)]
            
            # Crop image to bounding box
            img = cv2.imread(img_path)
            x1, y1, x2, y2 = map(int, largest_box)
            cropped = img[y1:y2, x1:x2]
            
            # Resize to 256x256
            resized = cv2.resize(cropped, (256, 256))
            
            # Save processed image
            output_file = os.path.join(output_path, os.path.basename(img_path))
            cv2.imwrite(output_file, resized)
    
    print(f"Preprocessed {len(image_paths)} images")

def clean_directory(directory):
    """Remove all files and subdirectories from the specified directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"Cleaned directory: {directory}")


def run_pipeline():
    """
    Webcam Image Capture
    """
    print("\n Step 1: Capturing images...")
    # Clean webcam_images directory before starting
    clean_directory(webcam_imgs_path)
    webcam = WebcamCapture(webcam_imgs_path)
    webcam.capture_images()

    """
    Preprocess Images
    """
    print("\n Step 1.1: Preprocessing images...")
    processed_imgs_path = os.path.join(webcam_imgs_path, "processed")
    os.makedirs(processed_imgs_path, exist_ok=True)
    preprocess_images(webcam_imgs_path, processed_imgs_path)


    """
    Image(-s) Classification
    """
    print("\n Step 2: Classifying images...")
    df = pd.read_csv(csv)
    brand_classes = sorted(df['brand'].unique())
    body_type_classes = sorted(df['body_style'].unique())


    # FIXME! Ferrari is not in the dataset, but it is in the weights file
    brand_classes.append("FERRARI")
    

    print("\nClass Mapping Verification:")
    print("Brand classes:", brand_classes)
    print("Body type classes:", body_type_classes)
    
    # Create separate classifiers for brand and body type
    brand_classifier = CarClassifier(
        model_name=model_name,
        model_path=weights_brand,
        brand_classes=brand_classes,
        classifier_type="brand"
    )

    body_classifier = CarClassifier(
        model_name=model_name,
        model_path=weights_body,
        body_type_classes=body_type_classes,
        classifier_type="body"
    )

    # Get predictions from both classifiers
    brand_result = brand_classifier.classify_folder(processed_imgs_path)
    body_result = body_classifier.classify_folder(processed_imgs_path)

    brand = brand_result['most_common_brand']
    car_type = body_result['most_common_body_type']

    # Get probabilities from first predictions
    brand_probs = brand_result['all_predictions'][0]['probabilities']
    body_probs = body_result['all_predictions'][0]['probabilities']

    # Calculate confidence scores
    brand_confidence = float(brand_probs.max()) * 100
    body_confidence = float(body_probs.max()) * 100

    print("\n=== Classification Results ===")
    print(f"Brand:     {brand:<15} (confidence: {brand_confidence:.1f}%)")
    print(f"Body Type: {car_type:<15} (confidence: {body_confidence:.1f}%)")
    print("===========================")

    print("\nTop 3 Brand Predictions:")
    brand_indices = np.argsort(brand_probs)[-3:][::-1]  # Get last 3 in descending order
    for idx in brand_indices:
        print(f"{brand_classes[idx]:<15}: {brand_probs[idx]*100:.1f}% (index: {idx})")
    
    print("\nTop 3 Body Type Predictions:")
    body_indices = (-body_probs).argsort()[:3]
    for idx in body_indices:
        print(f"{body_type_classes[idx]:<15}: {body_probs[idx]*100:.1f}%")
    print("===========================\n")

    """
    Article Agent
    """
    print("\n Step 3: Creating JSON... \n")
    agent_pipeline = AgentPipeline(brand, car_type)
    response = agent_pipeline()
    # save json object to a file
    with open(json_path, "w") as f:
        f.write(response)

    """
    Article Assembler
    """
    print("Step 4: Assembling article...")
    # Initialize the pipeline
    assembler = AssemblerPipeline(json_path, images_path, output_pdf_path)
    # Set up the environment (install LaTex, wkhtmltopdf, Pandoc)
    assembler.setup(py_reqs=False)
    # Run the pipeline
    assembler.run_pipeline()

if __name__ == "__main__":
    run_pipeline()
