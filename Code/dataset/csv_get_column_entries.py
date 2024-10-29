import csv

def extract_unique_entries(csv_file, column_name, output_file):
    # Create a set to store unique entries
    unique_entries = set()

    # Read the CSV file
    with open(csv_file, mode='r', newline='', encoding='ISO-8859-1') as file:
        reader = csv.DictReader(file)
        
        # Iterate over the rows to collect unique entries from the specified column
        for row in reader:
            entry = row.get(column_name)
            if entry:  # If the entry exists and is not None
                unique_entries.add(entry.strip())  # Add the entry to the set, remove leading/trailing spaces

    # Write the unique entries to a text file as a bullet point list
    with open(output_file, mode='w', encoding='utf-8') as file:
        for entry in sorted(unique_entries):  # Sorting to keep the list organized
            file.write(f"• {entry}\n")

    print(f"Unique entries from '{column_name}' saved to {output_file}")


# Example usage:
extract_unique_entries('/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_v1/final_2_lowercase.csv',
                       'body_style',
                       '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_v1/body_styles.txt')
