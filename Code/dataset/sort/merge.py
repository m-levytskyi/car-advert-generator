import pandas as pd

def merge_csvs(csv1_path, csv2_path, output_csv):
    # Read the CSV files
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Concatenate the DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to: {output_csv}")

# Example usage
csv1_path = "DS1_segmented_0.8conf_84%arranged.csv"
csv2_path = "DS2_segmented_brand_30_0.8conf_67%arranged.csv"
output_csv = "DS1+2_brand_0.8conf.csv"
merge_csvs(csv1_path, csv2_path, output_csv)
