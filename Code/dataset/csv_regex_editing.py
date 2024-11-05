import pandas as pd
import re

def apply_regex_to_csv(file_path, source_column, target_column, regex_pattern):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Compile the regular expression pattern
    pattern = re.compile(regex_pattern)
    
    # Apply the regular expression to each entry in the source column
    # and store the result in the target column as a string without brackets or extra characters
    df[target_column] = df[source_column].apply(
        lambda x: ' '.join(pattern.findall(str(x))) if pd.notnull(x) else None
    )
    
    # Save the modified DataFrame back to the same CSV file
    df.to_csv(file_path, index=False)
    print(f"Results written to {file_path}")

# Example usage
file_path = '/Users/johannesdecker/Downloads/reduced_dataset.csv'
source_column = 'path_to_jpg'
target_column = 'sub_path'
regex_pattern = r'([^/]+/[^/]+)$' # pattern captures the last two segments (after the second to last slash) of the path

apply_regex_to_csv(file_path, source_column, target_column, regex_pattern)
