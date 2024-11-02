import math
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as functional
import wandb
import time
import nvidia_smi
import dataloader
import importlib.util
import tqdm

spec = importlib.util.spec_from_file_location("AlexNet", 'alexnet/alexnet.py')
alexnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alexnet)

def validateModel(val_model,val_dataloader,amountToValidate):
    print("Evaluation ",end="",flush=True)
    startval = time.time()
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
            val_loss += model.loss(result,label)*label.size(0)

            _, predicted = torch.max(result, 1)
            correct+= (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            validated+=len(image)
            if(validated>=amountToValidate and amountToValidate >= 1):
                break
        
        wandb.log({f"val/img/Confusion Matrix {epoch}": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_labels,
            class_names=val_dataloader.dataset.classes,
            title=f"Confusion Matrix Epoch {epoch}"  )
            })

        val_metrics = {"val/val_loss": val_loss / validated,
                        "val/val_accuracy": correct / validated}
        wandb.log({val_metrics})
        print(f"ended in {(time.time()-startval)/60:.2f}m - Validation Loss: {val_loss / validated:3f}, Validation Accuracy: {correct / validated:.2f}")

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
    dataset_val = dataloader.CustomCarDataset(csv_file=csv,transform=transformations, phase='test', in_memory=in_ram_dataset,amount=640)

    if(modelname=="alexnet"):
        model =  alexnet.AlexNet(amount_classes=len(dataset_train.classes)).to(device)
    if(modelname=="resnet"):
      model =  alexnet.AlexNet(amount_classes=len(dataset_train.classes)).to(device)

    train_loader = DataLoader(dataset_train, batch_size=model.batchsize, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=model.batchsize, shuffle=False, num_workers=0,drop_last=True)
    
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
        model.train()
        steps = 0
        max_vram_use=0
        epochloss=0
        evaltrain=0
        
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", unit='batch') as tbar:
            for img, clss in tbar:
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

                tbar.set_postfix(loss=loss.item())

                steps+=1
                evaltrain+=len(img)

                metrics = {"train/train_loss": loss,
                        "train/epoch": epoch,
                        "train/example_ct": len(img)}
                if steps < math.ceil(train_data_size/model.batchsize):
                    wandb.log(metrics)       
        
        if(epoch % 10 == 0 or epoch==model.epochs-1):
            validateModel(model,val_loader,-1)

    if(device == "cuda"):
        nvidia_smi.nvmlShutdown()




   
