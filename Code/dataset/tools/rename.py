import os

def rename_files_recursively(directory):
    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(directory):
        # Get the name of the current directory
        parent_dir = os.path.basename(dirpath)
        
        # Iterate over each file in the current directory
        for filename in filenames:
            # Create the new name for the file with parent directory prefix
            new_filename = f"{parent_dir}_{filename}"
            
            # Full path of the current file
            old_file_path = os.path.join(dirpath, filename)
            
            # Full path of the new file
            new_file_path = os.path.join(dirpath, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

# Example usage:
directory = "/Users/johannesdecker/Downloads/Car_Models_3778_sorted/train"
rename_files_recursively(directory)

