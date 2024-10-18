import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as functional
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

class AlexNet(nn.Module):
    def __init__(self,amount_classes):
        super().__init__()

        self.step = nn.Sequential(
        # input: RGB -> 3 channels with format 227x227
        nn.Conv2d(3,96,11,4),
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
        nn.ReLU(),
        # amount_classes x 1 (1000x1 in alexnet paper)
        nn.Softmax(1),
        )

        #gaußian 0-distribution, bias 0 for every convolution
        for conv in self.step:
            if isinstance(conv, nn.Conv2d):
                nn.init.normal_(conv.weight,0,0.01)
                nn.init.constant_(conv.bias, 0)
        #bias for 2nd,4th and 5th convolution to 1
        nn.init.constant_(self.step[4].bias, 1)
        nn.init.constant_(self.step[10].bias, 1)
        nn.init.constant_(self.step[12].bias, 1)


    def forward(self,x):
        return self.step(x)


if __name__ == '__main__':

    #TODO: integrate own dataset (from filesystem)
    #TODO: crop pics to 256x256, extract 4 more patches (up-left, up-right, down-left, down-right) and mirror them horizontal, augment RGB values with PCA
    dataset = datasets.DTD("./dataset",download=True,
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()]))
    
    model = AlexNet(len(dataset.classes)).to(device)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=128)
    
    #use stochastic gradient descent as optimizer function
    optimizer = optim.SGD(params=model.parameters(),lr=0.01,weight_decay=0.0005,momentum=0.9)
    #reduce learning rate by factor 10 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    steps = 1
    for epoch in range(90):

        for img, clss in dataloader:
            # load images to gpu / cpu
            img, clss = img.to(device), clss.to(device)

            #call model (forward?)
            input=model(img)
            #loss
            loss = functional.cross_entropy(input,clss)

            #backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            steps+=1

        #save checkpoint to disk to possibly continue training
        checkpoint_path = os.path.join("./checkpoints", 'alexnet_states_epoch{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': steps,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }
        torch.save(state, checkpoint_path)

        break





   
