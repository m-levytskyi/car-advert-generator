import os
import pandas as pd

def delete_unlisted_files(directory, csv_file, column_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get the list of relative file paths from the specified column
    listed_files = df[column_name].tolist()
    
    # Make sure the paths are correctly formatted for comparison
    listed_files = [os.path.normpath(file) for file in listed_files]
    
    # Loop through all files in the specified directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the relative path
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            relative_path = os.path.normpath(relative_path)
            
            # Check if the file is not listed in the CSV file
            if relative_path not in listed_files:
                # Delete the file
                file_to_delete = os.path.join(root, file)
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted: {file_to_delete}")
                except Exception as e:
                    print(f"Failed to delete {file_to_delete}: {e}")

directory = "/Users/johannesdecker/Downloads/Car_Models_3778_sorted/train"  # Replace with your directory
csv_file = "/Users/johannesdecker/Downloads/reduced_dataset.csv"    # Replace with your CSV file
column_name = "sub_path"      # Replace with the name of the column containing file paths

delete_unlisted_files(directory, csv_file, column_name)
