import pandas as pd
import re

def process_column(csv_file, column_to_edit, new_column_name, output_file):
    try:
        # Load the CSV file with a specified encoding
        df = pd.read_csv(csv_file, encoding='ISO-8859-1', delimiter=';')
        
        # Process the column: convert to lowercase, remove special characters, and replace spaces with underscores
        df[new_column_name] = df[column_to_edit].str.lower()\
                                                 .str.replace(r'[^a-z0-9\s]', '', regex=True)\
                                                 .str.strip()\
                                                 .str.replace(r'\s+', '_', regex=True)
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Processed CSV saved as {output_file}")
    
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
csv_file = '/Users/johannesdecker/Downloads/reduced_dataset_2.csv'          # Input CSV file
column_to_edit = 'segment'    # Name of the column to process
new_column_name = 'segment_2' # Name of the new column with edited data
output_file = '/Users/johannesdecker/Downloads/reduced_dataset_3.csv'       # Output CSV file

process_column(csv_file, column_to_edit, new_column_name, output_file)
