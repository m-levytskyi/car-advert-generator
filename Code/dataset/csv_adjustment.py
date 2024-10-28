import csv
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def replace_substring_in_csv(input_file, output_file, old_substring, new_substring):
    # Detect the file encoding
    encoding = detect_encoding(input_file)
    print(f"Detected encoding: {encoding}")
    
    # Open the input CSV file for reading with detected encoding
    with open(input_file, 'r', newline='', encoding=encoding) as infile:
        reader = csv.reader(infile)
        
        # Open the output CSV file for writing with UTF-8 encoding
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Iterate over each row in the input CSV
            for row in reader:
                # Replace the old substring with the new one in every cell
                new_row = [cell.replace(old_substring, new_substring) for cell in row]
                # Write the modified row to the output CSV
                writer.writerow(new_row)

# Example usage
input_csv = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/Mappe1.csv'
output_csv = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/Mappe2.csv'
old_text = 'MercedesAMG'
new_text = 'MERCEDESBENZ'

replace_substring_in_csv(input_csv, output_csv, old_text, new_text)

print(f"Replaced '{old_text}' with '{new_text}' in {input_csv} and saved it as {output_csv}")
