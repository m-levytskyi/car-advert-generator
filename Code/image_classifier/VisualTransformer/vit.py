import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig
import torch.nn.functional as functional

class VisionTransformer(nn.Module):
    def __init__(self, amount_classes):
        super(VisionTransformer, self).__init__()
        
        # Load a pre-trained ViT model and add a classification head
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.num_labels = amount_classes
        self.model = ViTModel(config)
        
        # Classifier head
        self.classifier = nn.Linear(self.model.config.hidden_size, amount_classes)
        
        # Model parameters
        self.epochs = 50
        self.batchsize = 16
        self.loss = functional.cross_entropy
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
    def forward(self, x):
        outputs = self.model(x).last_hidden_state[:, 0]  # Extract the [CLS] token's embedding
        logits = self.classifier(outputs)
        return logits
