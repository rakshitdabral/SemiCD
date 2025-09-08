import os
import shutil

# Create a script to rename files
def rename_files():
    # Get all files from A folder (T1 images)
    a_files = os.listdir('test_enaroch/A')
    
    for a_file in a_files:
        # Extract the patch number (e.g., 01393 from patch_01393_T1.tif)
        patch_num = a_file.split('_')[1]
        
        # Create new names
        new_name = f"{patch_num}.tif"
        
        # Rename files
        os.rename(f"test_enaroch/A/{a_file}", f"test_enaroch/A/{new_name}")
        os.rename(f"test_enaroch/B/patch_{patch_num}_t2.tif", f"test_enaroch/B/{new_name}")
        # os.rename(f"added/mask/mask_{patch_num}.tif", f"added/mask/{new_name}")

if __name__ == "__main__":
    rename_files()