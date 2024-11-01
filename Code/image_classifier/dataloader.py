import pandas as pd
import os
from PIL import Image
from torchvision import  transforms
from torch.utils.data import Dataset

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
        all=len(self.data)
        print(f"{all} images")
        imagescounter=0

        for _, row in self.data.iterrows():
            path_to_img = os.path.join("../../",row['path_to_jpg'])

            if not os.path.exists(path_to_img):
                raise Exception(f"File not found: {path_to_img}")

            image = Image.open(path_to_img).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append((image, self.label_map[row['brand']]))

            imagescounter+=1
            print(f"image {imagescounter} from {all} with shape {image.shape} loaded")

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