import torch
from torchvision import transforms
from PIL import Image

class CarClassifier:
    def __init__(self, model_path, brand_classes, body_type_classes, device=None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.brand_classes = brand_classes
        self.body_type_classes = body_type_classes
        
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardnormalisierung
        ])
    
    def classify(self, image_path):

        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            brand_logits, body_type_logits = logits

            brand_probs = torch.nn.functional.softmax(brand_logits, dim=1)
            body_type_probs = torch.nn.functional.softmax(body_type_logits, dim=1)

            predicted_brand = torch.argmax(brand_probs, dim=1).item()
            predicted_body_type = torch.argmax(body_type_probs, dim=1).item()
        
        return {
            "brand": self.brand_classes[predicted_brand],
            "body_type": self.body_type_classes[predicted_body_type],
            "brand_probabilities": brand_probs.cpu().numpy(),
            "body_type_probabilities": body_type_probs.cpu().numpy()
        }
