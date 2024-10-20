import os

def rename_files_in_directory(directory_path):
    try:
        # Get a list of all files in the directory
        files = os.listdir(directory_path)
        
        # Loop through each file in the directory
        for file_name in files:
            # Construct the old file path
            old_file_path = os.path.join(directory_path, file_name)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(old_file_path):
                # Create the new file name with '1_' prepended
                new_file_name = f"1_{file_name}"
                
                # Construct the new file path
                new_file_path = os.path.join(directory_path, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
        
        print("Files renamed successfully!")
    
    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage
directory = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/IMAGES_OF_130k_Images_(512x512)_-_Universal_Image_Embeddings_cars"  # Replace with the actual directory path
rename_files_in_directory(directory)
