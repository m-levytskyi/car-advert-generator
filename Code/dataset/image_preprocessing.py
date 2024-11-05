import os
from PIL import Image
import torch
from torchvision import transforms

# Function to preprocess images in a directory and its subdirectories
def preprocess_images(input_dir, output_dir, file_extension=".jpg"):
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loop through all files in the directory and subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(file_extension.lower()):
                # Open image
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert("RGB")
                
                # Apply transformations
                img_tensor = preprocess(img)
                
                # Save the processed image tensor to output directory
                # Create equivalent output directory structure if it doesn't exist
                relative_path = os.path.relpath(img_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the tensor as a torch file
                torch.save(img_tensor, output_path.replace(file_extension, ".pt"))
                
                print(f"Processed and saved: {output_path.replace(file_extension, '.pt')}")

# Example usage
input_directory = "/Users/johannesdecker/Downloads/DS1_Car_Models_3778_sorted_256"  # Replace with the path to your images
output_directory = "/Users/johannesdecker/Downloads/DS1_Car_Models_3778_sorted_256_preprocessed"  # Replace with the path for saving processed images
file_ext = ".jpg"  # Define the image file extension to process

preprocess_images(input_directory, output_directory, file_ext)
