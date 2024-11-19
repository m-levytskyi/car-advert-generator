import os
import pandas as pd

def clean_csv_by_file_paths(csv_file_path, column_name, base_directory):
    """
    Cleans a CSV file by checking if the file paths in a specific column exist.
    Non-existing file paths result in the row being removed.

    Args:
        csv_file_path (str): The path to the CSV file.
        column_name (str): The column containing file paths.
        base_directory (str): The base directory for relative paths.

    Returns:
        int: The number of rows deleted.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")

    # Calculate the absolute paths
    df['absolute_path'] = df[column_name].apply(lambda x: os.path.join(base_directory, x))

    # Identify rows where the file exists
    initial_row_count = len(df)
    df = df[df['absolute_path'].apply(os.path.exists)]

    # Count the deleted rows
    final_row_count = len(df)
    deleted_row_count = initial_row_count - final_row_count

    # Drop the helper 'absolute_path' column
    df = df.drop(columns=['absolute_path'])

    # Save the cleaned DataFrame back to the CSV
    df.to_csv(csv_file_path, index=False)

    # Print the number of deleted rows
    print(f"Deleted {deleted_row_count} rows with non-existing file paths.")

    return deleted_row_count

# Usage example:
csv_file_path = '/Users/johannesdecker/Downloads/DS1_Car_Models_3778_sorted_256/reduced_dataset.csv'  # Replace with your CSV file path
column_name = 'sub_path'    # Replace with your column name
base_directory = '/Users/johannesdecker/Downloads/DS1_Car_Models_3778_sorted_256/'  # Replace with your base directory

try:
    clean_csv_by_file_paths(csv_file_path, column_name, base_directory)
except Exception as e:
    print(f"An error occurred: {e}")
