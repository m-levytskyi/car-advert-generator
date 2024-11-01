from torch import nn
import torch.optim as optim
import torch.nn.functional as functional

class AlexNet(nn.Module):
    def __init__(self,amount_classes):
        super().__init__()

        self.epochs=90
        self.batchsize=128
        self.loss=functional.cross_entropy

        self.step = nn.Sequential(
        # input: RGB -> 3 channels with format 227x227
        nn.Conv2d(3,96,11,4,padding=2), #padding -> added to support 224x224 input
        nn.ReLU(),
        nn.LocalResponseNorm(5,0.0001,0.75,2),
        ## 96x55x55
        nn.MaxPool2d(3,2),
        # 96x27x27
        nn.Conv2d(96,256,5,padding=2),
        nn.ReLU(),
        nn.LocalResponseNorm(5,0.0001,0.75,2),
        # 256x27x27
        nn.MaxPool2d(3,2),
        # 256x13x13
        nn.Conv2d(256,384,3,padding=1),
        nn.ReLU(),
        # 384x13x13
        nn.Conv2d(384,384,3,padding=1),
        nn.ReLU(),
        # 384x13x13
        nn.Conv2d(384,256,3,padding=1),
        nn.ReLU(),
        #256x13x13
        nn.MaxPool2d(3,2),
        # 256x6x6
        nn.Flatten(),
        # (256x6x6) -> 9216x1
        nn.Dropout(0.5),
        nn.Linear(9216,4096),
        nn.ReLU(),
        # 4096x1
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU(),
        # 4096x1
        nn.Linear(4096,amount_classes),
        # amount_classes x 1
        # cross-entropy has softmax built in
        # nn.Softmax(1),
        )

        # BIASES AND WEIGHTS CAUSE STAGNATION; STOP LEARNING PROCESS
        # PROPABLY ONLY USEFUL FOR IMAGENET DATASET LEARN PROCESS TUNING
        # for conv in self.step:
        #     if isinstance(conv, nn.Conv2d):
        #         nn.init.normal_(conv.weight,0,0.01)
        #         nn.init.constant_(conv.bias, 0)
        # #bias for 2nd,4th and 5th convolution to 1
        # nn.init.constant_(self.step[4].bias, 1)
        # nn.init.constant_(self.step[10].bias, 1)
        # nn.init.constant_(self.step[12].bias, 1)

        self.optimizer=optim.Adam(params=self.parameters(),lr=0.0001)

    def forward(self,x):
        return self.step(x)