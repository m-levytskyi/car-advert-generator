import pandas as pd
import os
from torchvision import  transforms
from torch.utils.data import Dataset
import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch

def load_single_image(row, label_map, tolabel_row, augmentindex):
    path_to_img = os.path.join("../../", row["path_to_jpg"])
    if not os.path.exists(path_to_img):
        raise Exception(f"File not found: {path_to_img}")

    image = cv2.imdecode(np.fromfile(path_to_img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.resize(image, (256, 256)) # images are preprocessed to this resolution
    h, w, _ = image.shape
    startx, starty = 0, 0

    if augmentindex == 0:
        # Top-left patch
        image = image[:224, :224]
    elif augmentindex == 1:
        # Top-left patch + mirror horizontal
        image = image[:224, :224][:, ::-1].copy()
    elif augmentindex == 2:
        # Top-right patch
        image = image[:224, -224:]
    elif augmentindex == 3:
        # Top-right patch + mirror horizontal
        image = image[:224, -224:][:, ::-1].copy()
    elif augmentindex == 4:
        # Bottom-left patch
        image = image[-224:, :224]
    elif augmentindex == 5:
        # Bottom-left patch + mirror horizontal
        image = image[-224:, :224][:, ::-1].copy()
    elif augmentindex == 6:
        # Bottom-right patch
        image = image[-224:, -224:]
    elif augmentindex == 7:
        # Bottom-right patch + mirror horizontal
        image = image[-224:, -224:][:, ::-1].copy()
    elif augmentindex == 8:
        # Middle patch
        startx = w // 2 - 112
        starty = h // 2 - 112
        image = image[starty:starty + 224, startx:startx + 224]
    elif augmentindex == 9:
        # Middle patch + mirror horizontal
        startx = w // 2 - 112
        starty = h // 2 - 112
        image = image[starty:starty + 224, startx:startx + 224][:, ::-1].copy()

    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)

    label = label_map[row[tolabel_row]]

    return image, label

class CustomCarDataset(Dataset):
    def __init__(self, csv_file, phase='train', in_memory=False,amount=0,tolabel='brand',equallydistributed=False,ignoreLabels=[],augmentation=False):
        self.data_full = pd.read_csv(csv_file)

        if len(ignoreLabels)>0:
            self.data_full = self.data_full[~self.data_full[tolabel].isin(ignoreLabels)]

        self.classes = sorted(self.data_full[tolabel].unique())
        print(f"{len(self.classes)} labels found from column {tolabel}")
        self.data = self.data_full[self.data_full['phase'] == phase]

        self.tolabelrow=tolabel
        self.equalDistribution=equallydistributed
        self.in_memory = in_memory
        self.augment = augmentation

        if equallydistributed:
            grouped = self.data.groupby(tolabel)
            min_count = grouped.size().min()
            self.data = grouped.apply(lambda x: x.sample(min_count)).reset_index(drop=True)

        if amount > 0 and not self.equalDistribution:
            self.data = self.data.sample(amount)

        if self.augment:
            self.in_memory=False
            print("Cant load images to RAM when augmentation is active because of excessive consumption!")

        self.label_map = {label: idx for idx, label in enumerate(self.classes)}

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
                    futures.append(executor.submit(load_single_image, row, self.label_map,self.tolabelrow,8))
                
                for future in futures:
                    image, label = future.result()
                    self.images.append((image, label))
                    tbar.update(1)

    def __len__(self):
        return len(self.data) if not self.augment else len(self.data)*10

    def __getitem__(self, idx):
        if self.in_memory:
            image, label = self.images[idx]
        else:
            if self.augment:
                index = idx % 10
                idx_div_10= int((idx - index) / 10)
                image, label = load_single_image(self.data.iloc[idx_div_10],self.label_map,self.tolabelrow,index)
            else:
                image, label = load_single_image(self.data.iloc[idx],self.label_map,self.tolabelrow,8)

        return image, label