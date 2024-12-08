import os
import shutil

def copy_files_with_extension(src_dir, dest_dir, file_extension):
    """
    Copies all files with a specific extension from the source directory
    (including its subdirectories) to the destination directory.

    Parameters:
    src_dir (str): The source directory.
    dest_dir (str): The destination directory.
    file_extension (str): The file extension to search for (e.g., ".txt").
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(file_extension):
                # Construct full file path
                full_file_path = os.path.join(root, file)
                # Copy the file to the destination directory
                shutil.copy(full_file_path, dest_dir)
                print(f"Copied: {full_file_path} -> {dest_dir}")

# Example usage
source_directory = '/Users/johannesdecker/Downloads/88,000+_Images_of_Cars' # '/path/to/source/directory'
destination_directory = '/Users/johannesdecker/Downloads/DS2' # '/path/to/destination/directory'
file_ext = '.jpg'  # Replace with the desired file extension

copy_files_with_extension(source_directory, destination_directory, file_ext)


