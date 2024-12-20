import pandas as pd
import random

def limit_classes(csv_path, output_csv, column_name="brand", limit=6489):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Group by the specified column and count the occurrences of each class
    brand_counts = df[column_name].value_counts()

    # Create an empty DataFrame to hold the limited rows
    limited_df = pd.DataFrame()

    # Iterate through each class
    for brand, count in brand_counts.items():
        # Get all rows for the current class
        brand_rows = df[df[column_name] == brand]
        
        if count > limit:
            # Randomly sample rows to meet the limit
            brand_rows = brand_rows.sample(n=limit, random_state=42)

        # Append the rows to the limited DataFrame
        limited_df = pd.concat([limited_df, brand_rows])

    # Save the limited DataFrame to a new CSV file
    limited_df.to_csv(output_csv, index=False)

# Example usage
csv_path = "../DS1+2_brand_0.8conf"
output_csv = f"{csv_path}_cut.csv"
csv_path=csv_path + ".csv"
limit_classes(csv_path, output_csv, column_name="brand", limit=5452)
