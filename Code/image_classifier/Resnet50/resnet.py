import torchvision
import torch.optim as optim
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self,amount_classes):
        super().__init__()

        self.epochs=50
        self.batchsize=64
        self.loss=nn.CrossEntropyLoss()
                
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, amount_classes)

        self.optimizer=optim.Adam(self.model.fc.parameters(), lr=0.001, weight_decay=0.0001)

    def forward(self,x):
        return self.model(x)