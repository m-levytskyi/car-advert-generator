"""Car Classification Pipeline - processes car images and generates articles"""

import os
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
import shutil
from ultralytics import YOLO
import argparse

from webcam.webcam_capture import WebcamCapture
from image_classifier.classify import CarClassifier
from article_agent.agent_pipeline import AgentPipeline 
from article_assembler.assembler_pipeline import AssemblerPipeline

# Configuration
PATHS = {
    'webcam': "Code/webcam/webcam_images",
    'demo_images': "Code/webcam/demo_images",
    'dataset': "Code/dataset/data/reduced_dataset_adjusted.csv",
    'json_output': "Code/article_agent/json/output.json",
    'images': "Code/article_assembler/tmp/imgs",
    'article': "Code/article.pdf",
    'weights_brand': "Code/image_classifier/alexnet/weights/Alexnet_brand_0.86acc.PTH",
    'weights_body': "Code/image_classifier/alexnet/weights/Alexnet_body_style_0.84acc.PTH"
}

MODEL_NAME = "alexnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_classes():

    #TODO : classes hardcoded

    """Load brand and body type classes from dataset"""
    df = pd.read_csv(PATHS['dataset'])
    brand_classes = sorted(df['brand'].unique())
    print(f'Brand classes: {brand_classes}')
    body_classes = sorted(df['body_style'].unique())
    print(f'Body classes: {body_classes}')
    return brand_classes, body_classes

def preprocess_images(input_path, output_path, confidence=0.8):

    # TODO: confidence

    """Preprocess images using YOLO detection"""
    print("Preprocessing images...")
    model = YOLO("yolo11x.pt")
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in Path(input_path).glob("*") 
                  if p.suffix.lower() in valid_extensions]
    
    processed = 0
    for img_path in image_paths:
        results = model(str(img_path), verbose=False)
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_indices = results[0].boxes.cls.cpu().numpy()
            confidence_scores = results[0].boxes.conf.cpu().numpy()
            
            # Filter for cars (class index 2) with confidence above threshold
            car_indices = np.where((class_indices == 2) & (confidence_scores > confidence))[0]

            if len(car_indices) > 0:
                areas = (boxes[car_indices, 2] - boxes[car_indices, 0]) * (boxes[car_indices, 3] - boxes[car_indices, 1])
                largest_car_idx = car_indices[np.argmax(areas)]

                # Get the bounding box for the largest car
                x1, y1, x2, y2 = map(int, boxes[largest_car_idx])

                # Load and process the image
                img = cv2.imread(str(img_path))
                cropped = img[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (256, 256))

                # Save the processed image
                output_file = os.path.join(output_path, img_path.name)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                cv2.imwrite(output_file, resized)
                processed += 1
    
    print(f"Preprocessed {processed} of {len(image_paths)} images")
    return processed

def clean_directory(directory):
    """Clean directory by removing all contents"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def classify_images(processed_path, brand_classes, body_classes):
    """Classify processed images"""
    brand_classifier = CarClassifier(
        model_name=MODEL_NAME,
        model_path=PATHS['weights_brand'],
        brand_classes=brand_classes,
        classifier_type="brand"
    )
    
    body_classifier = CarClassifier(
        model_name=MODEL_NAME,
        model_path=PATHS['weights_body'],
        body_type_classes=body_classes,
        classifier_type="body"
    )
    
    results = {
        'brand': brand_classifier.classify_folder(processed_path),
        'body': body_classifier.classify_folder(processed_path)
    }
    
    return {
        'brand': results['brand']['most_common_brand'],
        'body_type': results['body']['most_common_body_type'],
        'brand_confidence': float(results['brand']['all_predictions'][0]['probabilities'].max()) * 100,
        'body_confidence': float(results['body']['all_predictions'][0]['probabilities'].max()) * 100,
        'brand_probs': results['brand']['all_predictions'][0]['probabilities'],
        'body_probs': results['body']['all_predictions'][0]['probabilities']
    }

def print_results(predictions, brand_classes, body_classes):
    """Print classification results"""
    print("\n=== Classification Results ===")
    print(f"Brand:     {predictions['brand']:<15} ({predictions['brand_confidence']:.1f}%)")
    print(f"Body Type: {predictions['body_type']:<15} ({predictions['body_confidence']:.1f}%)")

def main():
    print("**** Loading. Please wait ****")
    
    parser = argparse.ArgumentParser(description="Car Classification Pipeline")
    parser.add_argument('--mode', choices=['camera', 'images'], default='camera', help="Mode to run the pipeline: 'camera' or 'images'")
    parser.add_argument('--images_path', type=str, default=PATHS['demo_images'], help="Path to the images to be used instead of webcam capture (required if mode is 'images')")
    args = parser.parse_args()

    # 1. Capture Images (or use a Demo Images Folder)
    if args.mode == 'camera':
        images_path = PATHS['webcam']

        print("\nStep 1: Capturing images...")
        clean_directory(PATHS['webcam'])
        processed_path = os.path.join(PATHS['webcam'], "processed")
        os.makedirs(processed_path, exist_ok=True)
        
        webcam = WebcamCapture(PATHS['webcam'])
        webcam.capture_images()   

    elif args.mode == 'images':
        if not args.images_path:
            parser.error("--images_path is required when mode is 'images'")
        images_path = args.images_path

        print(f"\nStep 1: Using images from {images_path}...")
        processed_path = os.path.join(images_path, "processed")
        os.makedirs(processed_path, exist_ok=True)
        clean_directory(processed_path)
    
    else:
        print("Error: Invalid mode. Please use 'camera' or 'images'")
        return

    
    # 2. Preprocess Images
    processed_count = preprocess_images(images_path, processed_path)
    if processed_count == 0:
        print("Error: No cars detected in images. Please try again.")
        return
        
    # 3. Classify Images
    brand_classes, body_classes = load_classes()
    predictions = classify_images(processed_path, brand_classes, body_classes)
    print_results(predictions, brand_classes, body_classes)
    
    # 4. Generate Article
    print("\nStep 4: Generating article...")
    agent = AgentPipeline(predictions['brand'], predictions['body_type'])
    response = agent()
    
    with open(PATHS['json_output'], "w") as f:
        f.write(response)
    
    # 5. Assemble Article
    print("\nStep 5: Assembling article...")
    assembler = AssemblerPipeline(
        PATHS['json_output'],
        PATHS['images'],
        PATHS['article']
    )
    assembler.setup(py_reqs=False)
    assembler.run_pipeline()

if __name__ == "__main__":
    main()