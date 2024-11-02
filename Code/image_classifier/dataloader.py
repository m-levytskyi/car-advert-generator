import pandas as pd
import os
from PIL import Image
from torchvision import  transforms
from torch.utils.data import Dataset
import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch

def load_single_image(row, label_map):
    path_to_img = os.path.join("../../", row.iloc[6])
    if not os.path.exists(path_to_img):
        raise Exception(f"File not found: {path_to_img}")

    image = cv2.imdecode(np.fromfile(path_to_img, dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256))
    h, w, _ = image.shape
    startx = w // 2 - (224 // 2)
    starty = h // 2 - (224 // 2)
    image = image[starty:starty + 224, startx:startx + 224]

    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)

    label = label_map[row.iloc[0]]

    return image, label

class CustomCarDataset(Dataset):
    def __init__(self, csv_file, phase='train', in_memory=False,amount=0):
        self.data_full = pd.read_csv(csv_file)
        self.classes = sorted(self.data_full['brand'].unique())
        self.data = self.data_full[self.data_full['phase'] == phase]

        if(amount>0):
            self.data=self.data.sample(amount)

        self.label_map = {label: idx for idx, label in enumerate(self.classes)}
        self.in_memory = in_memory

        if self.in_memory:
            self.images = []
            self.preload_images()

    def preload_images(self):
        self.images = []
        with ThreadPoolExecutor() as executor:
            rows = list(self.data.iterrows())
            futures = []
            with tqdm.tqdm(total=len(rows), desc="Images to RAM", unit='image', dynamic_ncols=True) as tbar:
                for _, row in rows:
                    futures.append(executor.submit(load_single_image, row, self.label_map))
                
                for future in futures:
                    image, label = future.result()
                    self.images.append((image, label))
                    tbar.update(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.in_memory:
            image, label = self.images[idx]
        else:
            image, label = load_single_image(self.data.iloc[idx],self.label_map)

        return image, label