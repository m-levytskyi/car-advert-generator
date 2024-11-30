import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_csv(file_path, columns_to_consider, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Check that all specified columns exist in the dataframe
    for col in columns_to_consider:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file.")

    # Plot histograms for each column and save the plots
    for col in columns_to_consider:
        plt.figure(figsize=(10, 6))
        value_counts = data[col].value_counts()
        percentages = (value_counts / len(data) * 100).round(2)
        
        bars = value_counts.plot(kind='bar')
        plt.title(f"Histogram of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        # Annotate bars with exact numbers and percentages, rotated 90 degrees
        for i, (count, pct) in enumerate(zip(value_counts, percentages)):
            plt.text(
                i, 
                count + max(value_counts) * 0.02,  # Slightly above the bar
                f"{count}\n({pct}%)", 
                ha='center', 
                va='bottom', 
                rotation=90
            )
        
        plt.tight_layout()
        output_path = os.path.join(output_directory, f"histogram_{col}.png")
        plt.savefig(output_path)
        plt.close()

    # Generate co-occurrence matrices and save the plots
    for i, col1 in enumerate(columns_to_consider):
        for col2 in columns_to_consider[i+1:]:
            co_occurrence = pd.crosstab(data[col1], data[col2])
            co_occurrence_percentage1 = co_occurrence.div(co_occurrence.sum(axis=1), axis=0) * 100
            co_occurrence_percentage2 = co_occurrence.div(co_occurrence.sum(axis=0), axis=1) * 100
            
            # Calculate dynamic figure size based on the number of rows and columns
            num_rows, num_cols = co_occurrence.shape
            fig_width = max(8, num_cols * 1.5)  # Scale width by number of columns
            fig_height = max(6, num_rows * 1.5)  # Scale height by number of rows

            plt.figure(figsize=(fig_width, fig_height))
            ax = sns.heatmap(co_occurrence, annot=False, fmt="d", cmap="Blues", cbar=False, linewidths=.5)
            plt.title(f"Co-Occurrence Matrix: '{col1}' vs '{col2}' (Counts, Row%, Col%)")
            plt.xlabel(col2)
            plt.ylabel(col1)

            # Annotate each cell with counts and percentages
            for y in range(co_occurrence.shape[0]):
                for x in range(co_occurrence.shape[1]):
                    count = co_occurrence.iloc[y, x]
                    pct_by_row = co_occurrence_percentage1.iloc[y, x]
                    pct_by_col = co_occurrence_percentage2.iloc[y, x]
                    ax.text(
                        x + 0.5, y + 0.5,
                        f"{count}\n{pct_by_row:.1f}%\n{pct_by_col:.1f}%",
                        ha='center', va='center',
                        color="black",
                        fontsize=min(12, max(8, 300 / max(num_rows, num_cols)))  # Adjust font size dynamically
                    )
            
            plt.tight_layout()
            output_path = os.path.join(output_directory, f"co_occurrence_{col1}_vs_{col2}_combined.png")
            plt.savefig(output_path)
            plt.close()

# Example usage
file_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/reduced_dataset_adjusted.csv"  # Replace with your CSV file path
columns_to_consider = ["brand", "body_style", "segment"]  # Replace with your column names
output_directory = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778_sorted_256/plots_output"  # Specify your output directory
analyze_csv(file_path, columns_to_consider, output_directory)
