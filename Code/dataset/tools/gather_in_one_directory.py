import os
import shutil

def organize_directory(main_directory):
    """
    Move all files from subdirectories to the main directory
    and delete all the subdirectories.

    :param main_directory: The path of the main directory to organize
    """
    # Ensure the main directory exists
    if not os.path.exists(main_directory):
        print(f"The directory '{main_directory}' does not exist.")
        return

    # Iterate over the contents of the main directory
    for root, dirs, files in os.walk(main_directory, topdown=False):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Move the file to the main directory
            try:
                shutil.move(file_path, main_directory)
            except Exception as e:
                print(f"Failed to move {file}: {e}")

        for dir in dirs:
            # Construct the full directory path
            dir_path = os.path.join(root, dir)
            # Remove the directory
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Failed to delete directory {dir}: {e}")

    print("Directory organization completed.")

if __name__ == "__main__":
    # Specify the main directory
    # main_directory = input("Enter the path of the directory to organize: ").strip()
    main_directory = "/Users/johannesdecker/Downloads/images.cv_33mdprbld2vsvfc8l3obl"
    organize_directory(main_directory)
