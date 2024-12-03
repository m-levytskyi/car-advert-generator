import pandas as pd

def convert_csv_to_lowercase(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)
    
    # Convert all string data to lowercase
    for column in df.columns:
        if df[column].dtype == object:  # Check if the column is of type object (usually strings)
            df[column] = df[column].str.lower()
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

# Example usage:
input_path = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/final_2.csv'
output_path = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/final_2_lowercase.csv'
convert_csv_to_lowercase(input_path, output_path)
