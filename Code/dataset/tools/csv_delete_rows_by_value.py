import pandas as pd

def remove_rows_by_column_value(input_csv, output_csv, column_name, value_to_remove):
    """
    Removes rows from a CSV file where the specified column contains a certain value.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to save the modified CSV file.
    :param column_name: Name of the column to filter on.
    :param value_to_remove: Value in the column that, if matched, will result in row removal.
    """
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(input_csv)
    
    # Filter out rows where the column equals the specified value
    filtered_df = df[df[column_name] != value_to_remove]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Rows with '{value_to_remove}' in column '{column_name}' have been removed. Saved to '{output_csv}'.")

# Example usage
if __name__ == "__main__":
    # Example: Removing rows from a CSV file
    input_file = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/reduced_dataset_body_style_adjusted.csv"
    output_file = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/reduced_dataset_5.csv"
    column_to_check = "brand"
    value_to_exclude = "FERRARI"
    
    # Call the function
    remove_rows_by_column_value(input_file, output_file, column_to_check, value_to_exclude)
