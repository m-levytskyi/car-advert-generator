import os
from PIL import Image
import datetime

def resize_image(image_path, new_size, log_file=None):
    """Resize the image to the specified size and overwrite the original."""
    try:
        with Image.open(image_path) as img:
            img = img.resize(new_size, Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing
            img.save(image_path)  # Overwrite the original file
            log_message = f"Resized and saved: {image_path}\n"
            print(log_message)
            if log_file:
                log_file.write(log_message)
    except Exception as e:
        log_message = f"Failed to resize {image_path}: {e}\n"
        print(log_message)
        if log_file:
            log_file.write(log_message)

def resize_images_in_directory(directory, new_size):
    """Walk through a directory and its subdirectories to resize images."""
    # Define supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Create log file with the name based on the directory and timestamp
    log_filename = f"{os.path.basename(directory.rstrip(os.sep))}_resize_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = os.path.join(directory, log_filename)

    with open(log_filepath, 'w') as log_file:
        log_message = f"Starting image resizing in directory: {directory}\n"
        print(log_message)
        log_file.write(log_message)

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    file_path = os.path.join(root, file)
                    resize_image(file_path, new_size, log_file)

        log_message = "Image resizing completed.\n"
        print(log_message)
        log_file.write(log_message)

if __name__ == "__main__":

    new_size = (256, 256)
    # Define the path to the directory with images
    directory = '/Users/johannesdecker/ADL_DATASETS/UNLABELED_IMAGES_ALL'
    # input("Enter the directory path to resize images: ")

    # Call the function to resize images and log output
    resize_images_in_directory(directory, new_size)

