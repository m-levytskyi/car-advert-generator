import math
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as functional
import os
import wandb
import time
import nvidia_smi
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import dataloader
import importlib.util

spec = importlib.util.spec_from_file_location("AlexNet", 'alexnet/alexnet.py')
alexnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alexnet)

def validateModel(val_model,val_dataloader,amountToValidate):
    val_model.eval()
    validated=0
    val_loss = 0.
    correct = 0.
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device) 
        
            result = val_model(image)
            val_loss += functional.cross_entropy(result,label)*label.size(0)

            _, predicted = torch.max(result, 1)
            correct+= (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            validated+=len(image)

            if(validated>=amountToValidate and amountToValidate >= 1):
                break
        
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    cax = ax.imshow(cm, cmap="Blues")
    tick_marks = np.arange(len(val_dataloader.dataset.classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(val_dataloader.dataset.classes, rotation=45, ha="right")
    ax.set_yticklabels(val_dataloader.dataset.classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.colorbar(cax)
    plt.close(fig)
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    wandb_img = wandb.Image(image_array, mode="RGBA", caption=f"Confusion Matrix Epoch {epoch}")
    wandb.log({f"val/img/Confusion Matrix {epoch}": wandb_img})

    print("of",validated,"images",end=" ",flush=True)
    return val_loss / validated , correct / validated

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    if(device == "cuda"):
        nvidia_smi.nvmlInit()
        nv_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    csv = "../dataset/sort/reduced_dataset.csv"    
    transformations=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()
                                        ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    in_ram_dataset=True
    modelname="alexnet"

    dataset_train = dataloader.CustomCarDataset(csv_file=csv,transform=transformations, phase='train', in_memory=in_ram_dataset)
    dataset_val = dataloader.CustomCarDataset(csv_file=csv,transform=transformations, phase='test', in_memory=in_ram_dataset,amount=128)

    if(modelname=="alexnet"):
        model =  alexnet.AlexNet(amount_classes=len(dataset_train.classes)).to(device)
    if(modelname=="resnet"):
      model =  alexnet.AlexNet(amount_classes=len(dataset_train.classes)).to(device)

    train_loader = DataLoader(dataset_train, batch_size=model.batchsize, shuffle=True, num_workers=0, pin_memory=True,drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=model.batchsize, shuffle=False, num_workers=0,pin_memory=True,drop_last=True)
    
    #wandb.login()
    wandb.init(
    project="Image classification",
    name="DTD_testing_w_Adam",
    config={
    "architecture": "AlexNet",
    "dataset": "DTD",
    "epochs": model.epochs,
    "batch_size": model.batchsize
    })

        
    train_data_size = len(train_loader.dataset)
    print("Train data contains", train_data_size,"images.")
    print("Validation data contains", len(val_loader.dataset),"images.")
    
    optimizer = model.optimizer

    for epoch in range(model.epochs):
        print("Epoch",epoch, "progress:",end=" ",flush=True)
        model.train()
        start = time.time()
        steps = 0
        max_vram_use=0
        epochloss=0
        evaltrain=0
        
        for img, clss in train_loader:
            img, clss = img.to(device), clss.to(device)
            
            if(device == "cuda"):
                nv_info = nvidia_smi.nvmlDeviceGetMemoryInfo(nv_handle)            
                if(max_vram_use<nv_info.used):
                    max_vram_use=nv_info.used

            output=model(img)
            loss = model.loss(output,clss)
            epochloss+=loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps+=1
            evaltrain+=len(img)

            metrics = {"train/train_loss": loss,
                       "train/epoch": epoch,
                       "train/example_ct": len(img)}
            if steps < math.ceil(train_data_size/model.batchsize):
                wandb.log(metrics)

            print(int(steps/(math.floor(train_data_size/model.batchsize))*100), "%",end=' ',flush=True,sep='')
        
        end = time.time()
        print("Epoch", epoch,"with",evaltrain ,"images finished in","%.2f"%((end-start)/60),"minutes with max","%.2f"%(max_vram_use / 1_000_000_000), "/", "%.2f"%(nv_info.total/ 1_000_000_000), "GB VRAM used - Train Loss:" ,"%.3f"%(epochloss/steps))
        
        if(epoch % 10 == 0):
            startval = time.time()
            print("Evaluation ",end="",flush=True)
            val_loss, accuracy = validateModel(model,val_loader,-1)
            val_metrics = {"val/val_loss": val_loss,
                        "val/val_accuracy": accuracy}
            wandb.log({**metrics, **val_metrics})
            print(f"ended in {(time.time()-startval)/60:.2f}m - Validation Loss: {val_loss:3f}, Validation Accuracy: {accuracy:.2f}")

    val_loss, accuracy = validateModel(model,val_loader,-1)
    val_metrics = {"val/val_loss": val_loss,
                        "val/val_accuracy": accuracy}
    wandb.log({**metrics, **val_metrics})
    print(f"ended in {(time.time()-startval)/60:.2f}m - Validation Loss: {val_loss:3f}, Validation Accuracy: {accuracy:.2f}")

    if(device == "cuda"):
        nvidia_smi.nvmlShutdown()




   
