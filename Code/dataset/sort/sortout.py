from ultralytics import YOLO
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class CustomCarDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            # Load image from disk
            row = self.data.iloc[idx]
            path_to_img = os.path.join("DS2_confidence_and_brand_selected/",row['image'])

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

    confidence = 0
    type="brand_30"
    csv_path = f"DS2_{type}.csv"
    output_file = f"DS2_segmented_{type}_{confidence}conf"
    output_folder = f"DS2_segmented_{type}_{confidence}conf/"

    csv = pd.read_csv(csv_path)
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
                    confidence_scores = output[index].boxes.conf
                    
                    # Get indices where the class is "car" (class index 2)
                    car_indices = (class_indices == 2).nonzero(as_tuple=True)[0]
                    
                    # Proceed if at least one "car" is detected with confidence > 0.8
                    car_indices = car_indices[confidence_scores[car_indices] > confidence]  # Filter indices with confidence > 0.8
                    if len(car_indices) > 0:
                        # Calculate areas of the bounding boxes
                        areas = [
                            (output[index].boxes.xywhn[i, 2] * img.shape[3]) *  # width
                            (output[index].boxes.xywhn[i, 3] * img.shape[2])   # height
                            for i in car_indices
                        ]
                        
                        # Find the index of the car with the largest bounding box
                        largest_car_idx = car_indices[areas.index(max(areas))]
                        
                        # Get the bounding box for the largest car
                        bounding_boxes = output[index].boxes.xywhn[largest_car_idx]
                        x_center, y_center, width, height = bounding_boxes[0], bounding_boxes[1], bounding_boxes[2], bounding_boxes[3]

                        # Convert normalized coordinates to absolute pixel coordinates
                        img_width, img_height = img.shape[3], img.shape[2]  # Shape: (batch, channels, height, width)
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        xmin = int(x_center - width / 2)
                        ymin = int(y_center - height / 2)
                        xmax = int(x_center + width / 2)
                        ymax = int(y_center + height / 2)

                        # Ensure bounding box coordinates are within image boundaries
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(img_width, xmax)
                        ymax = min(img_height, ymax)

                        # Convert the tensor image to a NumPy array
                        img_np = img[index].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                        img_np = (img_np * 255).astype(np.uint8)  # Scale pixel values to 0-255

                        # Crop and resize the bounding box region
                        cropped_img = img_np[ymin:ymax, xmin:xmax]  # Crop the region
                        resized_img = cv2.resize(cropped_img, (256, 256))  # Resize to 256x256

                        save_path = os.path.join(output_folder,csv.iloc[row[index]]['image'])
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, resized_img)

                        useful += 1
                        rows_new_csv.append(csv.iloc[row[index]])
                    else:
                        if manualEvaluation:
                            sorted_out.append(csv.iloc[row[index]])

            counter+=len(img)
            print(f"Processed {counter} from {all_images} images -> {"{:.2f}".format(counter/all_images*100)}%")

    print(f"{useful} of {all_images} kept -> {"{:.2f}".format(useful/all_images*100)}%")
    newcsv = pd.DataFrame(rows_new_csv)
    output_file = f"{output_file}_{int(useful/all_images*100)}%.csv"
    newcsv.to_csv(output_file, index=False)
    
    if(manualEvaluation):
        deleted = pd.DataFrame(sorted_out)
        for index, row in deleted.iterrows():
            image_path = os.path.join("../../../",row.iloc[6])

            if os.path.exists(image_path):
                print(image_path)
                image = Image.open(image_path)
                image.show()