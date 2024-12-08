import os

def rename_to_lowercase(directory):
    # Walk through each directory and subdirectory
    for root, dirs, files in os.walk(directory, topdown=False):
        # Rename files to lowercase
        for filename in files:
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, filename.lower())
            os.rename(old_path, new_path)
        
        # Rename directories to lowercase
        for dirname in dirs:
            old_path = os.path.join(root, dirname)
            new_path = os.path.join(root, dirname.lower())
            os.rename(old_path, new_path)

# Replace 'your_directory_path' with the path of the directory you want to modify
directory_path = '/Users/johannesdecker/Downloads/Car_Models_3778'
rename_to_lowercase(directory_path)

print("Renaming completed!")
