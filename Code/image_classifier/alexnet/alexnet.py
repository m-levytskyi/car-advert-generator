import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as functional
import os
import wandb
import time
import nvidia_smi
from sklearn.metrics import confusion_matrix

EPOCH_NUM=90
BATCH_SIZE=128

class AlexNet2(nn.Module):
    def __init__(self, amount_classes=1000):
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)

        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=amount_classes),

        )
        #self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
      #  print(x.shape)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class AlexNet(nn.Module):
    def __init__(self,amount_classes):
        super().__init__()

        self.step1 = nn.Sequential(
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
        #nn.Flatten(),
        #View((-1,9216)),
        )

        self.step2=nn.Sequential(
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

        #gaußian 0-distribution, bias 0 for every convolution
        # for conv in self.step:
        #     if isinstance(conv, nn.Conv2d):
        #         nn.init.normal_(conv.weight,0,0.01)
        #         nn.init.constant_(conv.bias, 0)
        # #bias for 2nd,4th and 5th convolution to 1
        # nn.init.constant_(self.step[4].bias, 1)
        # nn.init.constant_(self.step[10].bias, 1)
        # nn.init.constant_(self.step[12].bias, 1)

    def forward(self,x):
        x = self.step1(x)
        x.view(x.size(0), -1)
        #x = x.view(-1,9216)
        return self.step2(x)

def validateModel(val_model,val_dataloader,amountBatches):
    val_model.eval()
    evaluated=0
    val_loss = 0.
    correct = 0.
    with torch.inference_mode():
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device) 
        
        result = val_model(image)
        val_loss += functional.cross_entropy(result,label)*label.size(0)

        #firstLine = result.data[0]
        _, predicted = torch.max(result, 1)
        correct+= (predicted == label).sum()
        #arr = result.cpu().numpy()


        evaluated+=1
        if(evaluated>=amountBatches & amountBatches >= 1):
            return val_loss / ((len(image)) * evaluated) , correct #/ ((len(image)) * evaluated)
        
    return val_loss / ((len(image)) * evaluated) , correct / ((len(image)) * evaluated)

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    if(device == "cuda"):
        nvidia_smi.nvmlInit()
        nv_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    os.makedirs("./checkpoints", exist_ok=True)

    #wandb.login()
    wandb.init(
    project="Image classification",
    name="DTD_testing_w_Adam",
    config={
    "architecture": "AlexNet",
    "dataset": "DTD",
    "epochs": EPOCH_NUM,
    "batch_size":BATCH_SIZE
    })

    #TODO: integrate own dataset (from filesystem)
    #TODO: crop pics to 256x256, extract 4 more patches (up-left, up-right, down-left, down-right) and mirror them horizontal, augment RGB values with PCA

    dataset_train = datasets.DTD("./dataset",download=True,
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),)
    
    dataset_validate=datasets.DTD("./dataset",split="val",download=True,
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),)
    
    model =  AlexNet(amount_classes=len(dataset_train.classes)).to(device)
    #torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device) 
    
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        batch_size=BATCH_SIZE)
    
    dataloader_val = DataLoader(
        dataset_validate,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        batch_size=BATCH_SIZE)
        
    train_data_size = len(dataloader_train.dataset)
    print("Train data contains ", train_data_size," images.")
    print("Validation data contains ", len(dataloader_val.dataset)," images.")
    
    #use stochastic gradient descent as optimizer function
    optimizer = optim.Adam(params=model.parameters(),lr=0.0001)
    #reduce learning rate by factor 10 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(90):
        print("Epoch",epoch, "progress:",end=" ",flush=True)
        start = time.time()
        steps = 0
        max_vram_use=0
        epochloss=0
        
        for img, clss in dataloader_train:
            model.train()

            # load images to gpu / cpu
            img, clss = img.to(device), clss.to(device)
            
            nv_info = nvidia_smi.nvmlDeviceGetMemoryInfo(nv_handle)            
            if(max_vram_use<nv_info.used):
                max_vram_use=nv_info.used

            #call model (forward?)
            output=model(img)
            #loss
            loss = functional.cross_entropy(output,clss)
            epochloss+=loss

            #backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps+=1

            metrics = {"train/train_loss": loss,
                       "train/epoch": (steps + 1 + (math.ceil(train_data_size/BATCH_SIZE) * epoch)) / math.ceil(train_data_size/BATCH_SIZE),
                       "train/example_ct": len(img)}
            if steps < math.ceil(train_data_size/BATCH_SIZE):
                wandb.log(metrics)

            print(int(steps/(math.ceil(train_data_size/BATCH_SIZE))*100), "%",end=' ',flush=True,sep='')
        
        scheduler.step()
        print()
        end = time.time()
        print("Epoch", epoch, "finished in","%.2f"%((end-start)/60),"minutes with max","%.2f"%(max_vram_use / 1_000_000_000), "/", "%.2f"%(nv_info.total/ 1_000_000_000), "GB VRAM used.")
        
        startval = time.time()
        print("Evaluation ",end="",flush=True)
        val_loss, accuracy = validateModel(model,dataloader_train,1)
        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        print(f"ended in {(time.time()-startval)/60:.2f}m - Train Loss: {(epochloss/steps):.3f}, Validation Loss: {val_loss:3f}, Validation Accuracy: {accuracy:.2f}")

        #save checkpoint to disk to possibly continue training
        checkpoint_path = os.path.join("./checkpoints", 'alexnet_states_epoch{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': steps,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }
        torch.save(state, checkpoint_path)

    if(device == "cuda"):
        nvidia_smi.nvmlShutdown()




   
