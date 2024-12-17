import torch
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

from skimage.metrics import structural_similarity
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from functools import lru_cache


class CarInteriorDetector:
    def __init__(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.yolo = YOLO('yolov8x.pt')
        
    def analyze_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # Enhanced CLIP prompts
        texts = [
            # Positive (car interior specific)
            "close up photo of car dashboard with steering wheel and gauges",
            "car interior view showing dashboard and center console",
            "driver's perspective of car cockpit with instruments",
            "automotive interior with seats and dashboard controls",
            # Negative (strong exterior indicators)
            "full exterior view of car from outside",
            "car photographed in parking lot",
            "side view of automobile exterior",
            "car on road or street"
        ]

        # Debug CLIP scores
        inputs = self.clip_processor(images=image, text=texts, return_tensors="pt", padding=True)
        clip_scores = self.clip(**inputs).logits_per_image.softmax(dim=1)[0]
        
        # Print individual scores for debugging
        for text, score in zip(texts, clip_scores):
            print(f"CLIP score for '{text[:30]}...': {score.item():.4f}")

        # Calculate interior confidence using only positive prompts
        interior_confidence = clip_scores[:4].sum().item() 
        exterior_confidence = clip_scores[4:].sum().item()
        
        print(f"Interior confidence: {interior_confidence:.4f}")
        print(f"Exterior confidence: {exterior_confidence:.4f}")

        # Calculate confidence difference
        confidence_diff = interior_confidence - exterior_confidence
        print(f"Confidence difference (int-ext): {confidence_diff:.4f}")

        # Enhanced YOLO detection
        results = self.yolo(image)
        car_interior_objects = [
            'steering wheel', 'seat belt', 'dashboard', 'display', 
            'chair', 'speedometer', 'gear stick', 'car seat', 
            'radio', 'console', 'screen', 'meter', 'gauge'
        ]
        
        detected_objects = [results[0].names[int(obj)] for obj in results[0].boxes.cls]
        print(f"Detected objects: {detected_objects}")
        
        # Weighted object scoring
        #TODO update weights
        object_weights = {
            'steering wheel': 2.0,
            'dashboard': 2.0,
            'speedometer': 1.5,
            'gear stick': 1.5,
            'car seat': 1.0
        }
        
        object_score = sum(object_weights.get(obj, 0.5) 
                          for obj in detected_objects 
                          if obj in car_interior_objects)
        object_score = min(object_score / 5.0, 1.0)  # Normalize to 0-1
        
        print(f"Object detection score: {object_score:.4f}")

        # Adjusted final scoring with preference for interior indicators
        final_score = (0.2 * interior_confidence + 
                      0.2 * object_score + 
                      0.4 * confidence_diff +
                      0.2 * (1 - exterior_confidence))  # Penalize exterior confidence
        
        threshold = 0.45  # Adjusted threshold
        return final_score > threshold, final_score
    


class EnhancedCarInteriorDetector(CarInteriorDetector):
    def __init__(self):
        super().__init__()
        # Initialize DPT for depth estimation
        self.depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
        
    def analyze_image(self, image_path):
        # Get base CLIP+YOLO score
        base_result, base_score = super().analyze_image(image_path)
        print(f"Base (CLIP+YOLO) Score: {base_score}")
        
        # Load image once
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate additional scores
        depth_score = self._get_depth_score(image_rgb)
        print(f"Depth Score: {depth_score}")
        edge_score = self._get_edge_score(image)
        print(f"Edge Score: {edge_score}")
        color_score = self._get_color_score(image)
        print(f"Color Score: {color_score}")
        symmetry_score = self._get_symmetry_score(image)
        print(f"Symmetry Score: {symmetry_score}")
        blur_score = self._get_blur_score(image)
        print(f"Invert Blur Score: {1 - blur_score}")
        
        # Weighted combination
        enhanced_score = (
            0.55 * base_score +
            0.10 * depth_score +
            0.10 * edge_score +
            0.10 * color_score +
            0.05 * symmetry_score +
            0.10 * (1 - blur_score)  # Invert blur score
        )
        print(f"Enhanced Score: {enhanced_score}")
        
        return enhanced_score > 0.35, enhanced_score
    
    def _get_depth_score(self, image):
    # Uses DPT model to analyze 3D space patterns
    # Car interiors have distinct depth patterns:
    # - Dashboard close (high depth)
    # - Seats medium (medium depth)
    # - Rear window far (low depth)
        try:
            # Prepare image for depth estimation
            inputs = self.depth_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Normalize depth values
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = depth_map.numpy()
            
            # Calculate depth score based on typical car interior patterns
            depth_score = np.mean(depth_map) / np.max(depth_map)
            return depth_score
            
        except Exception as e:
            print(f"Depth analysis failed: {e}")
            return 0.5  # Return neutral score on failure
        

    def _get_edge_score(self, image):
    # Uses Canny edge detection
    # Car interiors have:
    # - Strong horizontal lines (dashboard)
    # - Strong vertical lines (center console)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges) / 255.0

    def _get_color_score(self, image):
    # Car interiors typically have:
    # - Lower saturation (grays, blacks)
    # - Controlled lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Car interiors often have darker, less saturated colors
        return 1 - (np.mean(hsv[:,:,1]) / 255.0)  # Inverse of saturation

    def _get_symmetry_score(self, image):
    # Car interiors are typically symmetric:
    # - Dashboard centered
    # - Controls evenly distributed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flipped = cv2.flip(gray, 1)
        score, _ = structural_similarity(gray, flipped, full=True)
        return score

    def _get_blur_score(self, image):
    # Car interior photos typically:
    # - Are taken with care
    # - Have good lighting
    # - Are sharp/focused
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(variance / 1000.0, 1.0)


def show_interior_images(
    dataset_path, 
    confidence_threshold=0.5,
    max_images=10, 
    output_dir=None,
    display=True
):
    """Process and display/save car interior images with confidence scores."""
    
    # Setup
    interior_images = []
    detector = EnhancedCarInteriorDetector()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


    # Process images
    img_count = 0
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                
                # Check if image is car interior
                print(f"\n Processing {image_path}")
                
                is_interior, confidence = detector.analyze_image(image_path)
                print(f"Interior image: {is_interior}")
                print(f"Confidence: {confidence} \n")

                img_count += 1
                print(f"\n Images processed: {img_count}")
                print(f"Interior images found: {len(interior_images)} of {max_images} \n")

                if is_interior:
                    interior_images.append((image_path, confidence))
                    print("Added to interior images")
                    
                    if len(interior_images) >= max_images:
                        break
                        
        if len(interior_images) >= max_images:
            break

    # Sort by confidence
    interior_images.sort(key=lambda x: x[1], reverse=True)
    
    # Display/save results
    for i, (img_path, confidence) in enumerate(interior_images):
        img = Image.open(img_path)
        
        if display:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Car Interior {i+1}\nConfidence: {confidence:.2f}\nPath: {img_path}")
            plt.axis('off')
            plt.show()
            
        if output_dir:
            basename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"{confidence:.2f}_{basename}")
            img.save(save_path)
            
    return interior_images

if __name__ == "__main__":
    # Example usage
    dataset_path = "Code/dataset/DATA/train"

    # "DATA/small"
    # 21 interior images + 21 exterior images

    results = show_interior_images(
        dataset_path=dataset_path,
        confidence_threshold=0.6,
        max_images=5,
        output_dir="Code/dataset/DATA/car_interiors_train",
        display=False
    )