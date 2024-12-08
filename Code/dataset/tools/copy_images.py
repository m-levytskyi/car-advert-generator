import os
import shutil

def copy_images(source_dir, destination_root):
    # Get the base name of the source directory
    base_name = os.path.basename(os.path.normpath(source_dir))
    
    # Create the new directory name that starts with 'IMAGES_OF_'
    destination_dir = os.path.join(destination_root, f'IMAGES_OF_{base_name}')
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created directory: {destination_dir}")
    
    # Loop through the source directory and all its subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file has a .jpg, .jpeg, or .png extension
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_dir, file)
                
                # Copy the image to the destination directory
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")
    
    print(f"All images have been copied to {destination_dir}")

# Example usage
source_directory = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/Stanford_Car_Dataset_by_classes_folder'  # Replace with your source directory
destination_root_directory = ''  # Replace with your destination root directory

copy_images(source_directory, destination_root_directory)
