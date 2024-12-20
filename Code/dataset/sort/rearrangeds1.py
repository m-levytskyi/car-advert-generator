import pandas as pd

def process_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    df = df[["brand", "body_style", "path_to_jpg", "phase"]]  # Rearrange columns

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to: {output_csv}")

# Example usage
input_csv = "DS1_segmented_0conf_100%"
output_csv = f"{input_csv}arranged.csv"
input_csv=input_csv+".csv"
process_csv(input_csv, output_csv)
