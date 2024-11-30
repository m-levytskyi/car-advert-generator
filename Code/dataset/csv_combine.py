import pandas as pd

def combine_entries_in_csv(file_path, column_name, entries_to_combine, new_entry, output_file_path):
    """
    Edits a CSV file by combining specific entries in a column and replacing them with a new entry.

    Parameters:
        file_path (str): The path to the input CSV file.
        column_name (str): The name of the column to edit.
        entries_to_combine (list): The list of entries to combine.
        new_entry (str): The new entry to replace the combined entries.
        output_file_path (str): The path to save the modified CSV file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Replace specified entries with the new entry
    df[column_name] = df[column_name].replace(entries_to_combine, new_entry)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"Updated CSV saved to {output_file_path}")

combine_entries_in_csv("/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/reduced_dataset_2.csv",
                       "body_style",
                       ["wagon", "van"],
                       "family_car",
                       "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/reduced_dataset_3.csv")

