import os
import pathlib

def replace_spaces_in_filenames(root_dir):
    """
    Recursively walks through a directory and replaces all spaces
    in file and directory names with an underscore '_'.
    
    It walks from the bottom up to safely rename directories
    after renaming their contents.
    """
    
    target_path = pathlib.Path(root_dir)
    if not target_path.is_dir():
        print(f"Error: Directory not found: {root_dir}")
        return

    print(f"Scanning for spaces in: {target_path.resolve()}")

    # Use os.walk with topdown=False to walk from the "bottom up"
    # This ensures we rename files *inside* a directory before
    # attempting to rename the directory itself.
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):


        # 2. Rename directories
        for dirname in dirnames:
            if 'madgwick_(al_borno)' in dirname:
                new_dirname = dirname.replace('madgwick_(al_borno)', 'madgwick (al borno)')
                original_dir = os.path.join(dirpath, dirname)
                new_dir = os.path.join(dirpath, new_dirname)
                
                try:
                    os.rename(original_dir, new_dir)
                    print(f"  Renamed dir:  {original_dir} -> {new_dir}")
                except OSError as e:
                    print(f"  Error renaming dir {original_dir}: {e}")
            if 'imu_data' in dirname:
                new_dirname = dirname.replace('imu_data', 'imu data')
                original_dir = os.path.join(dirpath, dirname)
                new_dir = os.path.join(dirpath, new_dirname)
                
                try:
                    os.rename(original_dir, new_dir)
                    print(f"  Renamed dir:  {original_dir} -> {new_dir}")
                except OSError as e:
                    print(f"  Error renaming dir {original_dir}: {e}")

if __name__ == "__main__":
    # --- PLEASE EDIT THIS ---
    # Set this to the directory you want to clean up
    TARGET_DIR = os.path.join("data")
    # ------------------------
    
    if not os.path.isdir(TARGET_DIR):
        print(f"Directory not found: '{TARGET_DIR}'")
        print("Please edit the TARGET_DIR variable in the script.")
    else:
        replace_spaces_in_filenames(TARGET_DIR)
        print("\n--- Space replacement complete. ---")