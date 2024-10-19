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

EPOCH_NUM=3
BATCH_SIZE=128

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
        # cross-entropy has softmax built in
        # nn.Softmax(1),
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

def validateModel(val_model,val_dataloader,amountBatches):
    val_model.eval()
    evaluated=0
    val_loss = 0.
    correct = 0.
    print("Evaluating...")
    with torch.inference_mode():
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device) 
        
        result = val_model(image)
        val_loss += functional.cross_entropy(result,label)*label.size(0)

        _, predicted = torch.max(result.data, 1)
        correct+= (predicted == label).sum().item()

        evaluated+=1
        if(evaluated>=amountBatches):
            return val_loss / (len(image)) * amountBatches , correct / (len(image)) * amountBatches

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
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()]))
    
    dataset_validate=datasets.DTD("./dataset",split="val",download=True,
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()]))
    
    model = AlexNet(len(dataset_train.classes)).to(device)

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        batch_size=128)
    
    dataloader_val = DataLoader(
        dataset_validate,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        batch_size=128)
        
    train_data_size = len(dataloader_train.dataset)
    print("Train data contains ", train_data_size," images.")
    print("Validation data contains ", len(dataloader_val.dataset)," images.")
    
    #use stochastic gradient descent as optimizer function
    optimizer = optim.Adam(params=model.parameters(),lr=0.0001)
    #reduce learning rate by factor 10 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(EPOCH_NUM):
        print("Epoch",epoch, "progress:")
        start = time.time()
        steps = 0
        max_vram_use=0

        model.train()
        
        for img, clss in dataloader_train:
            # load images to gpu / cpu
            img, clss = img.to(device), clss.to(device)
            
            nv_info = nvidia_smi.nvmlDeviceGetMemoryInfo(nv_handle)            
            if(max_vram_use<nv_info.used):
                max_vram_use=nv_info.used

            #call model (forward?)
            output=model(img)
            #loss
            loss = functional.cross_entropy(output,clss)

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
        
        val_loss, accuracy = validateModel(model,dataloader_val,1)
        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})

        print(f"Epoch: {epoch}, Train Loss: {loss:.3f}, Validation Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

        #save checkpoint to disk to possibly continue training
        checkpoint_path = os.path.join("./checkpoints", 'alexnet_states_epoch{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': steps,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }
        torch.save(state, checkpoint_path)

    nvidia_smi.nvmlShutdown()




   
