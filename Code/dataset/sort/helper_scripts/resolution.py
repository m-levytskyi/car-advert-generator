import os
from collections import Counter
import cv2
from tqdm import tqdm  # For the progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(file_path):
    """
    Processes a single image file to determine its resolution.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            # Get resolution (width, height)
            height, width = img.shape[:2]
            return (width, height)  # Return the resolution
        else:
            print(f"Could not read file: {file_path}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def collect_image_statistics_multithreaded(folder_path):
    """
    Collects image resolution statistics using multithreading.
    """
    # Dictionary to store resolution counts
    resolution_counter = Counter()

    # Supported file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png')

    # Gather all files to process
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                files_to_process.append(os.path.join(root, file))

    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks for all image files
        futures = {executor.submit(process_image, file): file for file in files_to_process}

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            resolution = future.result()
            if resolution:
                resolution_counter[resolution] += 1

    # Print filtered statistics
    print("\nImage Resolution Statistics (only showing resolutions with at least 20 files):")
    filtered_resolutions = {res: count for res, count in resolution_counter.items() if count >= 500}
    for resolution, count in filtered_resolutions.items():
        print(f"Resolution {resolution}: {count} files")

# Usage example
folder_path = "DS2_confidence_and_brand_selected"
collect_image_statistics_multithreaded(folder_path)