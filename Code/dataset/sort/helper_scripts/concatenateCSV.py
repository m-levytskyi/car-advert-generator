import pandas as pd

def concatenate_csv(file1, file2, output_file):
    """
    Reads two CSV files with the same columns and concatenates them into one CSV file.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path where the concatenated CSV file will be saved.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Concatenate the DataFrames
        concatenated_df = pd.concat([df1, df2], ignore_index=True)
        
        # Save the concatenated DataFrame to a new CSV file
        concatenated_df.to_csv(output_file, index=False)
        print(f"Concatenated CSV saved to: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
file1 = "file1.csv"
file2 = "file2.csv"
output_file = "concatenated.csv"
concatenate_csv(file1, file2, output_file)
