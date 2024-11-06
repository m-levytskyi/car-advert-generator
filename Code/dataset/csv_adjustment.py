import csv
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def replace_substrings_in_csv(input_file, output_file, replacements):
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
                # Replace each old substring with the new one in every cell
                new_row = [cell for cell in row]
                for old_substring, new_substring in replacements.items():
                    new_row = [cell.replace(old_substring, new_substring) for cell in new_row]
                # Write the modified row to the output CSV
                writer.writerow(new_row)

# Example usage
input_csv = '/Users/johannesdecker/Downloads/reduced_dataset_1.csv'
output_csv = '/Users/johannesdecker/Downloads/reduced_dataset_2.csv'
replacements = {
    "Roadster & Convertible": "convertible",
    "Compact": "compact",
    "Compact SUV": "compact_suv",
    "Coupe": "coupe",
    "Coupe Cabrio": "cabrio",
    "Crossover": "crossover",
    "Entry Premium": "entry_premium",
    "Exotic": "exotic",
    "Fullsize Pickup": "fullsize_pickup",
    "Heavy Duty Pickup": "heavy_duty_pickup",
    "Large": "large",
    "Large MPV": "large_mpv",
    "Large SUV": "large_suv",
    "Lower Premium": "lower_premium",
    "Luxury": "luxury",
    "Medium": "medium",
    "Medium MPV": "medium_mpv",
    "Medium Premium": "medium_premium",
    "Medium SUV": "medium_suv",
    "Midsize Pickup": "midsize_pickup",
    "Mini": "mini",
    "Premium Coupe": "premium_coupe",
    "Premium SUV": "premium_suv",
    "Small": "small",
    "Small MPV": "small_mpv",
    "Small Pickup": "small_pickup",
    "Small SUV": "small_suv",
    "Upper Premium": "upper_premium"
}

replace_substrings_in_csv(input_csv, output_csv, replacements)

print(f"Replaced strings in {input_csv} and saved the result as {output_csv}")
