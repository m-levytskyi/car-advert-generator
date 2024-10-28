import os

def get_filenames_without_extension(directory):
    """
    Get a set of filenames without extensions from a directory.
    """
    filenames = set()
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            filenames.add(os.path.splitext(file)[0])  # Extract filename without extension
    return filenames

def delete_files_with_specific_filenames(directory, filenames_to_delete):
    """
    Delete files from a directory where their name (without extension) matches the provided set.
    """
    for file in os.listdir(directory):
        file_name_without_extension = os.path.splitext(file)[0]
        if file_name_without_extension in filenames_to_delete:
            file_path = os.path.join(directory, file)
            print(f"Deleting file: {file_path}")  # Optionally print the file to be deleted
            os.remove(file_path)

def compare_and_delete(directory1, directory2):
    """
    Compare two directories and delete files that exist only in one directory based on their filename (without extension).
    """
    # Get filenames without extensions from both directories
    filenames_dir1 = get_filenames_without_extension(directory1)
    filenames_dir2 = get_filenames_without_extension(directory2)

    # Find files that exist only in one directory
    files_only_in_dir1 = filenames_dir1 - filenames_dir2
    files_only_in_dir2 = filenames_dir2 - filenames_dir1

    # Delete files that are only in one directory
    delete_files_with_specific_filenames(directory1, files_only_in_dir1)
    delete_files_with_specific_filenames(directory2, files_only_in_dir2)

# Example usage:
directory1 = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS2/images'
directory2 = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS2/labels'

compare_and_delete(directory1, directory2)
