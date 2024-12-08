import pandas as pd
import re

def replace_phrases_in_column(input_csv, output_csv, column_name, replacements):
    """
    Replaces phrases in a specified column of a CSV file based on a dictionary of replacements.

    Parameters:
        input_csv (str): The path to the input CSV file.
        output_csv (str): The path to save the modified CSV file.
        column_name (str): The name of the column where phrases should be replaced.
        replacements (dict): A dictionary where keys are phrases to replace, and values are the replacements.
    """

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    
    # Create a regular expression pattern to replace phrases
    def escape_special_chars(phrase):
        return re.escape(phrase)  # Escape special characters in the phrase

    # Construct the replacements with word boundaries and escaped special characters
    replacements_regex = {rf'\b{escape_special_chars(key)}\b': value for key, value in replacements.items()}
    
    # Replace phrases in the specified column using regex
    df[column_name] = df[column_name].replace(replacements_regex, regex=True)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Replacements complete. Modified file saved as '{output_csv}'.")

# Example usage
input_csv = '/Users/johannesdecker/Downloads/reduced_dataset_1.csv'  # Path to your input CSV file
output_csv = '/Users/johannesdecker/Downloads/reduced_dataset_2.csv'  # Path to save the output CSV file
column_name = 'segment'  # Name of the column with phrases to replace

replacements = {  # Dictionary of phrases to replace
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

# Call the function
replace_phrases_in_column(input_csv, output_csv, column_name, replacements)
