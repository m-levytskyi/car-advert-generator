import os
import pandas as pd

def compare_subdirectories_with_csv_column(directory_path, csv_path, column_name):
    # Step 1: List all subdirectories in the specified directory
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    # Step 2: Load the CSV file and extract the specified column
    try:
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the CSV file.")
            return
        csv_entries = df[column_name].astype(str).tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Step 3: Find subdirectories with no match in the CSV and vice versa
    unmatched_subdirs = [subdir for subdir in subdirectories if subdir not in csv_entries]
    unmatched_csv_entries = [entry for entry in csv_entries if entry not in subdirectories]

    # Step 4: Print findings
    print("\nSubdirectories with no match in CSV column:")
    if unmatched_subdirs:
        for subdir in unmatched_subdirs:
            print(f" - {subdir}")
    else:
        print("All subdirectories have a match in the CSV column.")

    print("\nCSV entries with no match in subdirectories:")
    if unmatched_csv_entries:
        for entry in unmatched_csv_entries:
            print(f" - {entry}")
    else:
        print("All CSV entries have a match in the subdirectories.")

# Usage example (replace with actual paths and column name)
directory_path = '/Users/johannesdecker/Downloads/Car_Models_3778/all_data'
csv_path = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_256/final_2_lowercase.csv'
column_name = 'path'

compare_subdirectories_with_csv_column(directory_path, csv_path, column_name)
