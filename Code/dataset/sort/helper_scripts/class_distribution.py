import pandas as pd
import matplotlib.pyplot as plt

def plot_brand_occurrences(csv_path, output_graph,class_type):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Count occurrences of each class in the 'brand' column
    brand_counts = df[class_type].value_counts()

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    bars = brand_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel(class_type, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Occurrences of each {class_type}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add exact numbers to each bar
    for i, count in enumerate(brand_counts):
        plt.text(i, count + max(brand_counts) * 0.01, str(count), ha='center', fontsize=10)

    # Save the graph to a file
    plt.tight_layout()
    plt.savefig(output_graph)
    plt.show()

# Example usage
csv_path = "../DS1+2_brand_0conf.csv"
output_graph = csv_path+ ".png"
plot_brand_occurrences(csv_path, output_graph,"brand")
