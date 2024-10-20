import os
import shutil
import math

def restructure_directory(source_dir, files_per_batch):
    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Directory '{source_dir}' does not exist.")
        return
    
    # Get list of files in the directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Check if there are files to process
    if not files:
        print("No files found in the directory.")
        return
    
    # Calculate the number of batches required
    num_files = len(files)
    num_batches = math.ceil(num_files / files_per_batch)

    # Create batch directories and move files
    for batch_num in range(1, num_batches + 1):
        batch_dir = os.path.join(source_dir, f'batch_{batch_num}')
        os.makedirs(batch_dir, exist_ok=True)
        
        # Move files to the current batch directory
        start_idx = (batch_num - 1) * files_per_batch
        end_idx = min(start_idx + files_per_batch, num_files)
        files_to_move = files[start_idx:end_idx]
        
        for file in files_to_move:
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(batch_dir, file)
            shutil.move(source_path, dest_path)
        
        print(f"Batch {batch_num} created with {len(files_to_move)} files.")

# Usage
source_directory = '/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/UNLABELED_IMAGES_ALL'  # Replace with your directory path
files_per_batch = 1000  # Replace with the number of files per batch

restructure_directory(source_directory, files_per_batch)
