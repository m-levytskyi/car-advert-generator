import torch
from torch.utils.data import DataLoader
import time
import dataloader
import importlib.util

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
        print(f"ended in {(time.time()-startval)/60:.2f}m - Validation Loss: {val_loss / validated:3f}, Validation Accuracy: {correct / validated:.2f}")

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

csv = "../dataset/sort/reduced_dataset.csv"    
in_ram_dataset=True
modelname="alexnet"

dataset_val = dataloader.CustomCarDataset(csv_file=csv, phase='test', in_memory=in_ram_dataset,amount=640*5)
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=0,drop_last=True,pin_memory=True)

if(modelname=="alexnet"):
   model =  alexnet.AlexNet(amount_classes=len(dataset_val.classes)).to(device)
if(modelname=="resnet"):
    model =  resnet.Resnet(amount_classes=len(dataset_val.classes)).to(device)
if(modelname=="visiontransformer"):
    model = visiontransformer.VisionTransformer(amount_classes=len(dataset_val.classes)).to(device)

checkpoint_path = f"checkpoints/alexnet_ep1/alexnet_epoch89_loss0.06238642707467079_weights.pth"

model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

validateModel(model,val_loader,-1)