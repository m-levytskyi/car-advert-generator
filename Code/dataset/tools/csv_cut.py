import pandas as pd
import chardet

# Function to detect the encoding of a file
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read a portion of the file to analyze encoding
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to edit the CSV file
def edit_csv(input_file, output_file, column_name, threshold):
    try:
        # Detect the encoding of the input file
        encoding = detect_encoding(input_file)
        print(f"Detected encoding: {encoding}")

        # Load the CSV file into a DataFrame using the detected encoding and specifying the delimiter
        df = pd.read_csv(input_file, encoding=encoding, delimiter=';', on_bad_lines='skip')  # Skip bad lines

        # Print the column names for debugging
        print("Available columns:", df.columns)

        # Strip any leading/trailing spaces from the column names
        df.columns = df.columns.str.strip()

        # Ensure the specified column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Count the occurrences of each value in the specified column
        value_counts = df[column_name].value_counts()

        # Keep only rows where the count of the value is greater than or equal to the threshold
        filtered_df = df[df[column_name].map(value_counts) >= threshold]

        # Save the filtered DataFrame to a new CSV file
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered CSV has been saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/Mappe2.csv'  # path to the input CSV file
output_file = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/Mappe2.csv'  # path where the output CSV will be saved
column_name = 'brand'  # name of the column to analyze
threshold = 50  # minimum number of occurrences

edit_csv(input_file, output_file, column_name, threshold)
