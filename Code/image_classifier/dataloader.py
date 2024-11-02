import pandas as pd
import os
from PIL import Image
from torchvision import  transforms
from torch.utils.data import Dataset
import tqdm
import psutil
import time

class CustomCarDataset(Dataset):
    def __init__(self, csv_file, transform=transforms.ToTensor(), phase='train', in_memory=False,amount=0):
        self.data_full = pd.read_csv(csv_file)
        self.classes = sorted(self.data_full['brand'].unique())
        self.data = self.data_full[self.data_full['phase'] == phase]

        if(amount>0):
            self.data=self.data.sample(amount)

        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.classes)}
        self.in_memory = in_memory

        if self.in_memory:
            self.images = []
            self.preload_images()

    def preload_images(self):
        """Load all images into memory."""
        with tqdm.tqdm(self.data.iterrows(), desc="Images to RAM", unit='image',total=len(self.data), dynamic_ncols=True) as tbar:
            for index, row in tbar:
                path_to_img = os.path.join("../../",row['path_to_jpg'])

                if not os.path.exists(path_to_img):
                    raise Exception(f"File not found: {path_to_img}")

                image = Image.open(path_to_img).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.images.append((image, self.label_map[row['brand']]))
                if(index % 100 == 0)
                    tbar.set_postfix(free_ram=psutil.virtual_memory()[4]/1073741824)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.in_memory:
            image, label = self.images[idx]
        else:
            row = self.data.iloc[idx]
            path_to_img = os.path.join("../../",row['path_to_jpg'])

            if not os.path.exists(path_to_img):
                raise Exception(f"File not found: {path_to_img}")

            image = Image.open(path_to_img).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.label_map[row['brand']]

        return image, label