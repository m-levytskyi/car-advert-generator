import os
import shutil

def compare_and_copy_dirs(dir1, dir2):
    """
    Compare subdirectories of dir1 and dir2. Copy contents from dir2 to dir1 if subdirectory names match.
    Print the names of subdirectories in dir2 that are not in dir1.
    """
    # List of subdirectories in dir1 and dir2
    dir1_subdirs = {name for name in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, name))}
    dir2_subdirs = {name for name in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, name))}
    
    # Identify matching and non-matching subdirectories
    matching_subdirs = dir1_subdirs.intersection(dir2_subdirs)
    non_matching_subdirs = dir2_subdirs - dir1_subdirs

    # Copy files from matching subdirectories in dir2 to dir1
    for subdir in matching_subdirs:
        src = os.path.join(dir2, subdir)
        dst = os.path.join(dir1, subdir)
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            if os.path.isfile(src_item):
                shutil.copy2(src_item, dst_item)
            elif os.path.isdir(src_item):
                if not os.path.exists(dst_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    # Merge directories if dst_item exists
                    for root, _, files in os.walk(src_item):
                        rel_path = os.path.relpath(root, src_item)
                        target_dir = os.path.join(dst_item, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        for file in files:
                            shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))

    # Print subdirectories that are in dir2 but not in dir1
    if non_matching_subdirs:
        print("Subdirectories in dir2 but not in dir1:")
        for subdir in non_matching_subdirs:
            print(subdir)

# Example usage
dir1 = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/all_data"
dir2 = "/Users/johannesdecker/adl-gruppe-1/Code/dataset/data/DS1_Car_Models_3778/test"
compare_and_copy_dirs(dir1, dir2)
