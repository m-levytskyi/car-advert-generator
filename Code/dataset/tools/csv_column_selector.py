import pandas as pd
import os

def edit_csv(input_file: str, output_file: str, columns_to_keep: list):
    """
    Edit a CSV file to keep only specified columns and reorder them.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path where the modified CSV file will be saved.
        columns_to_keep (list): List of column names to keep and the order they should appear.

    Returns:
        None
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {e}")

    # Verify that the required columns exist in the CSV
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing in the input CSV: {missing_columns}")

    # Keep only the required columns and reorder them
    df_filtered = df[columns_to_keep]

    # Save the modified CSV file
    try:
        df_filtered.to_csv(output_file, index=False)
        print(f"File has been saved successfully at: {output_file}")
    except Exception as e:
        raise Exception(f"Error saving the CSV file: {e}")


# Example usage:
if __name__ == "__main__":
    # Path to the input CSV file
    input_csv_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1+DS2_brand_70/DS1+DS2_brand_70.csv"

    # Path to save the modified CSV file
    output_csv_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1+DS2_brand_70/DS1+DS2_brand_70_cleaned.csv"

    # List of columns to keep and their order
    columns_to_keep = ["brand", "body_style", "path_to_jpg", "phase"]

    # Run the function
    try:
        edit_csv(input_csv_path, output_csv_path, columns_to_keep)
    except Exception as error:
        print(f"An error occurred: {error}")
