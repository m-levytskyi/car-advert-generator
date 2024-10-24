import pandas as pd
import os
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader


# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CustomCarDataset():
    def __init__(self, csv_file, is_pruned=False, path_to_train_folder='/Users/simonhampp/Desktop/WS2425/ADL/adl-gruppe-1/Code/image_classifier/Resnet50/Data/train', transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode
        self.train_folder = path_to_train_folder
        self.labels = sorted(self.data['brand'].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        
        if not is_pruned:
            # Refactor data to have one row per image file
            self.refactor_data()
            # Remove invalid rows during initialization
            self._clean_invalid_rows()
            self.data.to_csv('data_relevant_pruned.csv', index=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_dir = row['dir_path']
        image_file = row['image_file_name']  # Single file name after refactor
        correct_path = self._reformat_path(os.path.join(image_dir, image_file))

        # Check if path exists
        if not os.path.exists(correct_path):
            print(f"File not found: {correct_path}. Removing row {idx}.")
            self.data.drop(idx, inplace=True)  # Remove row if file doesn't exist
            self.data.reset_index(drop=True, inplace=True)
            return None  # Skip returning the item

        # Load the image
        image = Image.open(correct_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get the label for the image
        label = self.label_map[row['brand']]

        return image, label

    def _reformat_path(self, old_path):
        parts = old_path.split('/')
        
        # Extract necessary parts
        brand = parts[1]
        model = parts[2]
        year_range = parts[3]
        file_name = parts[4]

        # Create the new path
        new_path = os.path.join(f"{model}_{year_range}", file_name)

        return os.path.join(self.train_folder, new_path)

    def _clean_invalid_rows(self):
        # Go through all rows and remove those that have invalid paths
        valid_indices = []
        for idx, row in self.data.iterrows():
            image_dir = row['dir_path']
            image_file = row['image_file_name']  # Single file name after refactor
            correct_path = self._reformat_path(os.path.join(image_dir, image_file))

            if os.path.exists(correct_path):
                valid_indices.append(idx)
        
        # Keep only valid rows
        self.data = self.data.loc[valid_indices].reset_index(drop=True)

    def refactor_data(self):
        # Create a new DataFrame to store the split rows
        refactored_rows = []
        
        for _, row in self.data.iterrows():
            image_files = eval(row['image_file_names'])  # Convert string representation of list to actual list
            for image_file in image_files:
                new_row = row.copy()  # Copy the current row to modify
                new_row['image_file_name'] = image_file  # Create a single image file name column
                refactored_rows.append(new_row)

        # Create the refactored DataFrame
        self.data = pd.DataFrame(refactored_rows)
        self.data.reset_index(drop=True, inplace=True)


def prepare_datasets_and_dataloaders(csv_file, path_image_folder, data_transforms, csv_pruned = True):
    # Create datasets for train and validation
    train_dataset = CustomCarDataset(csv_file=csv_file, path_to_train_folder=path_image_folder, is_pruned=csv_pruned, transform=data_transforms['train'], mode='train')
    val_dataset = CustomCarDataset(csv_file=csv_file, path_to_train_folder=path_image_folder, is_pruned=csv_pruned, transform=data_transforms['val'], mode='val')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Store dataloaders in a dictionary for easy access
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Dataset sizes
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    return dataloaders, dataset_sizes, train_dataset