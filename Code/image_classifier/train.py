import math
import torch
from torch.utils.data import DataLoader
import wandb
import time
import dataloader
import importlib.util
import tqdm
import os

#import alexnet
spec = importlib.util.spec_from_file_location("AlexNet", 'alexnet/alexnet.py')
alexnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alexnet)

#import resnet
spec = importlib.util.spec_from_file_location("Resnet", 'Resnet50/resnet.py')
resnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet)

#import visiontransformer
spec = importlib.util.spec_from_file_location("VisionTransformer", 'VisualTransformer/vit.py')
visiontransformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visiontransformer)

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

        wandb.log({
        "val/val_loss": val_loss / validated,
        "val/val_accuracy": correct / validated
        })

        print(f"ended in {(time.time()-startval)/60:.2f}m - Validation Loss: {val_loss / validated:3f}, Validation Accuracy: {correct / validated:.2f}")

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # TODO: parse / config file for these arguments
    csv = "../dataset/sort/reduced_dataset.csv"    
    in_ram_dataset=True
    modelname="alexnet"

    checkpointdir= f"checkpoints/"           
    os.makedirs(checkpointdir, exist_ok=True)
    run_number = 1
    while True:
        run_dir = os.path.join("checkpoints", f"run{run_number}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            break
        run_number += 1

    dataset_train = dataloader.CustomCarDataset(csv_file=csv, phase='train', in_memory=in_ram_dataset)
    dataset_val = dataloader.CustomCarDataset(csv_file=csv, phase='test', in_memory=in_ram_dataset,amount=640)

    if(modelname=="alexnet"):
        model =  alexnet.AlexNet(amount_classes=len(dataset_train.classes)).to(device)
    if(modelname=="resnet"):
      model =  resnet.Resnet(amount_classes=len(dataset_train.classes)).to(device)
    if(modelname=="visiontransformer"):
        model = visiontransformer.VisionTransformer(amount_classes=len(dataset_train.classes)).to(device)

    train_loader = DataLoader(dataset_train, batch_size=model.batchsize, shuffle=True, num_workers=0, drop_last=True,pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=model.batchsize, shuffle=False, num_workers=0,drop_last=True,pin_memory=True)
    
    #wandb.login()
    wandb.init(
    project="Image classification evaluation on DS1",
    name=f"{modelname}_LR:{model.optimizer.param_groups[0]['lr']}_Epochs:{model.epochs}_BatchSize:{model.batchsize}_
    Loss:{model.loss.__name__}_Optimizer:{type(model.optimizer).__name__}_Classes:{len(dataset_train.classes)}",
    config={
    "architecture": modelname,
    "dataset": "DS1",
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
        epochloss=0
        
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", unit='batch') as tbar:
            for img, clss in tbar:
                img, clss = img.to(device), clss.to(device)

                output=model(img)
                loss = model.loss(output,clss)
                epochloss+=loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tbar.set_postfix(loss=loss.item())

                steps+=1

                metrics = {"train/train_loss": loss,
                        "train/epoch": epoch,
                        "train/example_ct": len(img)}
                if steps < math.ceil(train_data_size/model.batchsize):
                    wandb.log(metrics)       
        
        validateModel(model,val_loader,-1)
        if(epoch % 5 == 0 or epoch==model.epochs-1):
            torch.save(model.state_dict(), f"{checkpointdir}run{run_number}/{modelname}_epoch{epoch}_loss{epochloss/steps}_weights.pth")



   
