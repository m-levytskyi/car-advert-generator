import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_csvs_in_directory(directory_path, columns_to_consider, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # List to store data from all files
    all_data = []
    
    # Loop through all CSV files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            data = pd.read_csv(file_path)
            
            # Ensure all specified columns exist
            missing_cols = [col for col in columns_to_consider if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in file {file_name}.")
            
            # Append data from specified columns
            all_data.append(data[columns_to_consider])
    
    # Concatenate all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Plot histograms for each column and save the plots
    for col in columns_to_consider:
        plt.figure(figsize=(10, 6))
        value_counts = combined_data[col].value_counts()
        percentages = (value_counts / len(combined_data) * 100).round(2)
        
        bars = value_counts.plot(kind='bar')
        plt.title(f"Histogram of '{col}' (Combined Data)")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        # Annotate bars with exact numbers and percentages
        for i, (count, pct) in enumerate(zip(value_counts, percentages)):
            plt.text(
                i, 
                count + max(value_counts) * 0.02, 
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
            co_occurrence = pd.crosstab(combined_data[col1], combined_data[col2])
            co_occurrence_percentage1 = co_occurrence.div(co_occurrence.sum(axis=1), axis=0) * 100
            co_occurrence_percentage2 = co_occurrence.div(co_occurrence.sum(axis=0), axis=1) * 100
            
            # Calculate dynamic figure size based on the number of rows and columns
            num_rows, num_cols = co_occurrence.shape
            fig_width = max(8, num_cols * 1.5)
            fig_height = max(6, num_rows * 1.5)

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
                        fontsize=min(12, max(8, 300 / max(num_rows, num_cols)))
                    )
            
            plt.tight_layout()
            output_path = os.path.join(output_directory, f"co_occurrence_{col1}_vs_{col2}_combined.png")
            plt.savefig(output_path)
            plt.close()

# Example usage
directory_path = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data"  # Replace with your CSV files directory
columns_to_consider = ["brand", "body_style"]  # Replace with your column names
output_directory = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1+DS2+DS3_plots_output"  # Specify your output directory
analyze_csvs_in_directory(directory_path, columns_to_consider, output_directory)
