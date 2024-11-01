from ultralytics import YOLO
import torch
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from contextlib import contextmanager
import logging

class CustomCarDataset(Dataset):
    def __init__(self, csv_file, transform, in_memory=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = sorted(self.data['brand'].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.in_memory = in_memory

        # If in_memory is True, preload all images
        if self.in_memory:
            self.images = []
            self.preload_images()

    def preload_images(self):
        """Load all images into memory."""
        all=len(self.data)
        print(f"{all} images")
        imagescounter=0
        for idx, row in self.data.iterrows():
            path_to_img = os.path.join("../../../",row['path_to_jpg'])

            # Check if the image file exists
            if not os.path.exists(path_to_img):
                raise Exception(f"File not found: {path_to_img}")

            # Load and store the image
            image = Image.open(path_to_img).convert('RGB')
            if self.transform:
                image = self.transform(image)
            imagescounter+=1
            print(f"image {imagescounter} from {all} with shape {image.shape} loaded")
            self.images.append((image, self.label_map[row['brand']]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.in_memory:
            # If in_memory is True, return preloaded image
            image, label = self.images[idx]
        else:
            # Load image from disk
            row = self.data.iloc[idx]
            path_to_img = os.path.join("../../../",row['path_to_jpg'])

            # Check if the image file exists
            if not os.path.exists(path_to_img):
                raise Exception(f"File not found: {path_to_img}")

            # Load the image
            image = Image.open(path_to_img).convert('RGB')

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            # Get the label
            label = idx

        return image, label

if __name__ == '__main__':
    manualEvaluation=False # toggle manually to see "bad" images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model = YOLO("yolo11x.pt")

    csv_path = "../Data/DS1_vorläufig_Car_Models_3778/correct.csv"
    csv = pd.read_csv(csv_path)
    output_file = "reduced_dataset.csv"

    dataset = CustomCarDataset(csv_file=csv_path, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    all_images=len(loader.dataset)
    useful=0

    rows_new_csv = []
    sorted_out= []

    counter=0
    for img, row in loader:
            img = img.to(device)
            
            output = model(img, verbose=False)

            row = row.tolist()
            for index in range(len(output)):
                if output[index].boxes is not None and len(output[index].boxes) > 0:
                    class_indices = output[index].boxes.cls.long()
                    if (class_indices == 2).any():
                        useful+=1
                        rows_new_csv.append(csv.iloc[row[index]])
                    else:
                        if(manualEvaluation):
                            sorted_out.append(csv.iloc[row[index]])

            counter+=len(img)
            print(f"Processed {counter} from {all_images} images -> {"{:.2f}".format(counter/all_images*100)}%")

    print(f"{useful} of {all_images} kept -> {"{:.2f}".format(useful/all_images)}%")
    newcsv = pd.DataFrame(rows_new_csv)
    newcsv.to_csv(output_file, index=False)
    
    if(manualEvaluation):
        deleted = pd.DataFrame(sorted_out)
        for index, row in deleted.iterrows():
            image_path = os.path.join("../../../",row.iloc[6])

            if os.path.exists(image_path):
                print(image_path)
                image = Image.open(image_path)
                image.show()