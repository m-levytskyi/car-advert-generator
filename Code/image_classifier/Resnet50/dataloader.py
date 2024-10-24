import pandas as pd
import os
from PIL import Image
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
    def __init__(self, csv_file, transform=None, phase='train'):
        self.data_full = pd.read_csv(csv_file)
        self.data = self.data_full[self.data_full['phase'] == phase]
        self.transform = transform
        self.labels = sorted(self.data['brand'].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_img = row['path_to_jpg']

        # Check if path exists
        if not os.path.exists(path_to_img):
            raise Exception(f"File not found: {path_to_img}")

        # Load the image
        image = Image.open(path_to_img).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get the label for the image
        label = self.label_map[row['brand']]
        return image, label


def prepare_datasets_and_dataloaders(csv_file = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/correct.csv'):
    # Create datasets for train and validation
    train_dataset = CustomCarDataset(csv_file=csv_file, transform=data_transforms['train'])
    val_dataset = CustomCarDataset(csv_file=csv_file, transform=data_transforms['val'])

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

def get_test_dataloader(csv_file = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/correct.csv'):
    # Create datasets for train and validation
    test_dataset = CustomCarDataset(csv_file=csv_file, transform=data_transforms['train'], phase='test')

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    return test_loader, len(test_dataset), test_dataset

def create_correct_df(full_csv, img_folder):
    og_df = pd.read_csv(full_csv)
    new_df = pd.DataFrame()
    for idx, row in og_df.iterrows():
        folder_path = os.path.join(img_folder, row['path'])
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    new_row = row.copy()
                    new_row['path_to_jpg'] = os.path.join(folder_path, file)
                    new_df = new_df._append(new_row, ignore_index=True)
    return new_df

def create_full_csv(create_correct_df):
    """
    Create a full csv file with all the correct paths to the images.
    The final_2.csv file contains the paths to the images in the test and train folders but not the paths to the images themselves.
    """
    path_to_incomplete_csv = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/final_2.csv'
    path_to_img_test = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/test'
    path_to_img_train = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/train'

    test_csv = create_correct_df(path_to_incomplete_csv, path_to_img_test)
    train_csv = create_correct_df(path_to_incomplete_csv, path_to_img_train)

    # add column phase to csv with either test or train
    test_csv['phase'] = 'test'
    train_csv['phase'] = 'train'

    # merge the two dataframes
    full_csv = pd.concat([test_csv, train_csv])

    # save the full csv
    full_csv.to_csv('Code/dataset/Data/DS1_vorläufig_Car_Models_3778/correct.csv', index=False)

if __name__ == '__main__':
    # create_full_csv(create_correct_df)
    # dataloaders, dataset_sizes, train_dataset = prepare_datasets_and_dataloaders(data_transforms)
    path_to_correct_csv = 'Code/dataset/Data/DS1_vorläufig_Car_Models_3778/correct.csv'
    # Test the CustomCarDataset class
    dataset = CustomCarDataset(csv_file=path_to_correct_csv, transform=data_transforms['train'], phase='test')
    print(f"Number of samples in the dataset: {len(dataset)}")
    print(f"Number of unique labels in the dataset: {len(dataset.labels)}")
    print(f"Label map: {dataset.label_map}")
    # test getitem
    image, label = dataset[5]
    print(f"Image shape: {image}, Label: {label}")
