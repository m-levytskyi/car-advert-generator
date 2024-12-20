import pandas as pd

def process_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Drop the specified columns
    df.drop(columns=["confidence_body_styles", "confidence_brands"], inplace=True)

    # Modify the 'image' column by prepending the specified path
    base_path = "Code/dataset/Data/DS2_confidence_and_brand_selected/"
    df["image"] = base_path + df["image"]

    # Rename the column "image" to "path_to_jpg"
    df.rename(columns={"image": "path_to_jpg"}, inplace=True)

    # Add the "phase" column and rearrange the order of columns
    df["phase"] = "train"  # Add the new column with a constant value
    df = df[["brand", "body_style", "path_to_jpg", "phase"]]  # Rearrange columns

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to: {output_csv}")

# Example usage
input_csv = "DS2_segmented_brand_30_0conf_81%"
output_csv = f"{input_csv}arranged.csv"
input_csv=input_csv+".csv"
process_csv(input_csv, output_csv)
