import os
import shutil

def move_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Walk through the input folder and its subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)  # Full path to the file
            
            # Generate a unique destination path
            dest_path = os.path.join(output_folder, file)
            base_name, ext = os.path.splitext(file)
            counter = 1

            # If file exists, generate a new unique name
            while os.path.exists(dest_path):
                dest_path = os.path.join(output_folder, f"{base_name}_{counter}{ext}")
                counter += 1

            # Move the file
            shutil.move(file_path, dest_path)
            print(f"Moved: {file_path} -> {dest_path}")

# Example usage
input_folder = "gymgzmlaymrfcv7pxx4qg"
output_folder = "DS2_confidence_and_brand_selected"
move_files(input_folder, output_folder)
