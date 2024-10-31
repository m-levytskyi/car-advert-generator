import os
import shutil
import pandas as pd
import Levenshtein as lev

def delete_non_matching_subdirectories(csv_file, directory, column_name, distance_threshold=6):
    # Read the .csv file
    df = pd.read_csv(csv_file)
    
    # Get the list of subdirectories in the specified directory
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    # Get the list of valid entries from the specific column in the .csv file
    csv_entries = df[column_name].astype(str).tolist()
    
    # Set to hold deleted subdirectories and unmatched csv entries
    deleted_subdirectories = []
    unmatched_csv_entries = []

    # Function to find the closest matching entry based on Levenshtein distance
    def find_closest_match(subdir, csv_entries, threshold):
        for entry in csv_entries:
            if lev.distance(subdir, entry) <= threshold:
                return entry
        return None

    # Delete subdirectories that do not have a matching entry within the distance threshold
    for subdir in subdirectories:
        match = find_closest_match(subdir, csv_entries, distance_threshold)
        if not match:
            # Delete the subdirectory if no match is found within the distance threshold
            subdir_path = os.path.join(directory, subdir)
            shutil.rmtree(subdir_path)
            deleted_subdirectories.append(subdir)
    
    # Find unmatched entries in the csv file (entries for which no close match exists in subdirectories)
    for entry in csv_entries:
        match = find_closest_match(entry, subdirectories, distance_threshold)
        if not match:
            unmatched_csv_entries.append(entry)
    
    # Print results to the console
    if deleted_subdirectories:
        print(f"Deleted Subdirectories:")
        for subdir in deleted_subdirectories:
            print(subdir)
    else:
        print("No subdirectories were deleted.")

    if unmatched_csv_entries:
        print(f"\nEntries in CSV with no matching subdirectory:")
        for entry in unmatched_csv_entries:
            print(entry)
    else:
        print("All CSV entries had matching subdirectories.")

# Example usage
csv_file = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_256/final_2_lowercase.csv'  # Path to your .csv file
directory = '/Users/johannesdecker/Downloads/Car_Models_3778/all_data'  # Path to your directory containing subdirectories
column_name = 'path'  # The column name from the .csv file to compare against subdirectory names

delete_non_matching_subdirectories(csv_file, directory, column_name)
