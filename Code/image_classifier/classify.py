import os
import importlib
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from collections import Counter

class CarClassifier:
    def __init__(self, model_name, model_path, brand_classes, body_type_classes, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.brand_classes = brand_classes
        self.body_type_classes = body_type_classes
        self.total_classes = len(brand_classes) + len(body_type_classes)

        self.model = self._load_model(model_name, model_path)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_name, model_path):
        model_map = {
            "alexnet": ('Code/image_classifier/alexnet/alexnet.py', "AlexNet"),
            "resnet": ('Code/image_classifier/Resnet50/resnet.py', "Resnet"),
            "vit": ('Code/image_classifier/VisualTransformer/vit.py', "VisionTransformer")
        }

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        model_file, class_name = model_map[model_name]

        spec = importlib.util.spec_from_file_location(class_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_class = getattr(module, class_name)

        model = model_class(amount_classes=self.total_classes)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)

        return model

    def classify(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Split probabilities for brands and body types
            brand_probs = probs[0, :len(self.brand_classes)]
            body_probs = probs[0, len(self.brand_classes):]
            
            predicted_brand = torch.argmax(brand_probs).item()
            predicted_body = torch.argmax(body_probs).item()
        
        return {
            "brand": self.brand_classes[predicted_brand],
            "brand_probabilities": brand_probs.cpu().numpy(),
            "body_type": self.body_type_classes[predicted_body],
            "body_type_probabilities": body_probs.cpu().numpy()
        }

    def classify_folder(self, folder_path):
        """
        Classifies all images in a folder.
        
        Parameters:
        - folder_path (str): Path to the folder containing images.

        Returns:
        - dict: Most common body type in the folder.
        """
        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = [str(path) for path in Path(folder_path).glob("*") if path.suffix.lower() in valid_extensions]

        if not image_paths:
            raise ValueError(f"No valid image files found in the folder: {folder_path}")

        predictions = []
        for image_path in image_paths:
            try:
                pred = self.classify(image_path)
                predictions.append(pred)
            except Exception as e:
                print(f"Error classifying {image_path}: {e}")
        
        return self.aggregate_predictions(predictions)

    def aggregate_predictions(self, predictions):
        brand_types = [pred["brand"] for pred in predictions]
        body_types = [pred["body_type"] for pred in predictions]
        
        most_common_brand = Counter(brand_types).most_common(1)[0][0] if brand_types else None
        most_common_body = Counter(body_types).most_common(1)[0][0] if body_types else None
        
        return {
            "most_common_brand": most_common_brand,
            "most_common_body_type": most_common_body,
            "all_predictions": predictions
        }
