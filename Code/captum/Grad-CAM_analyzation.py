import torch
from torch.utils.data import DataLoader
import time
import dataloader
import importlib.util

device = torch.device("mps" if torch.backends.mps.is_available() else("cuda" if torch.cuda.is_available() else "cpu"))

#import alexnet
spec = importlib.util.spec_from_file_location("AlexNet", '../image_classifier/alexnet/alexnet.py')
alexnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alexnet)

#import resnet
spec = importlib.util.spec_from_file_location("Resnet", '../image_classifier/Resnet50/resnet.py')
resnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet)

#import visiontransformer
spec = importlib.util.spec_from_file_location("VisionTransformer", '../image_classifier/VisualTransformer/vit.py')
visiontransformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visiontransformer)

csv = "/DS1_Car_Models_3778_sorted_256_adjusted/reduced_dataset_adjusted.csv"
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

checkpoint_path = f"/trained_models/alexnet_epoch15_bestValLoss.pth"

model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

