import pandas as pd

def process_column(csv_file, column_to_edit, new_column_name, output_file):
    try:
        # Load the CSV file with a specified encoding
        df = pd.read_csv(csv_file, encoding='ISO-8859-1', delimiter=';')
        
        # Process the column: remove trailing whitespace, replace spaces with underscores
        df[new_column_name] = df[column_to_edit].str.strip().str.replace(r'\s+', '_', regex=True)
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Processed CSV saved as {output_file}")
    
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
csv_file = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/final.csv'          # Input CSV file
column_to_edit = 'title'    # Name of the column to process
new_column_name = 'path' # Name of the new column with edited data
output_file = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/final_2.csv'       # Output CSV file

process_column(csv_file, column_to_edit, new_column_name, output_file)
