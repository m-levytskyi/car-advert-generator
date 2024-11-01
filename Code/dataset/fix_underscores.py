import os
import re

def normalize_name(name):
    # Replace multiple underscores with a single underscore and convert to lowercase
    name = re.sub(r'__+', '_', name)
    return name.lower()

def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # Rename files
        for filename in files:
            new_filename = normalize_name(filename)
            original_file_path = os.path.join(root, filename)
            new_file_path = os.path.join(root, new_filename)
            if original_file_path != new_file_path:
                os.rename(original_file_path, new_file_path)
                print(f"Renamed file: {original_file_path} -> {new_file_path}")
        
        # Rename directories
        for dirname in dirs:
            new_dirname = normalize_name(dirname)
            original_dir_path = os.path.join(root, dirname)
            new_dir_path = os.path.join(root, new_dirname)
            if original_dir_path != new_dir_path:
                os.rename(original_dir_path, new_dir_path)
                print(f"Renamed directory: {original_dir_path} -> {new_dir_path}")

if __name__ == "__main__":
    directory = input("Enter the path to the directory to normalize: ")
    rename_files_in_directory(directory)
 