import pandas as pd
import os

def edit_csv(file_path, column_name, threshold, condition, save_path):
    """
    Edit a CSV file by deleting rows based on a threshold value for a specific column.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - column_name (str): The name of the column to apply the threshold filter.
    - threshold (float): The threshold value.
    - condition (str): Condition for filtering ('greater' or 'less').
    - save_path (str): Path to save the edited CSV file.

    Returns:
    - None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the CSV file.")
            return

        # Apply the threshold filter
        if condition == 'greater':
            filtered_df = df[df[column_name] <= threshold]
        elif condition == 'less':
            filtered_df = df[df[column_name] >= threshold]
        else:
            print("Invalid condition. Use 'greater' or 'less'.")
            return

        # Ensure the directory for save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the edited DataFrame to a new CSV file
        filtered_df.to_csv(save_path, index=False)
        print(f"Edited CSV file saved to: {save_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define file paths and parameters
file_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1+DS2_all_confidences.csv"   # Path to the input CSV file
save_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1+DS2_body_style_50/DS1+DS2_body_style_50.csv" # Path to save the edited CSV file
column_name = "confidence_body_styles" # Column to sort and filter by
threshold = 0.5 # Threshold value
condition = 'less' # Condition: 'greater' or 'less'

edit_csv(file_path, column_name, threshold, condition, save_path)
