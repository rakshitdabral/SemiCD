import os
import shutil

# Create a script to rename files
def rename_files():
    # Get all files from A folder (T1 images)
    a_files = os.listdir('cartosat/A')
    
    for a_file in a_files:
        # Extract the patch number (e.g., 01393 from patch_01393_T1.tif)
        patch_num = a_file.split('_')[1]
        
        # Create new names
        new_name = f"{patch_num}.tif"
        
        # Rename files
        os.rename(f"cartosat/A/{a_file}", f"cartosat/A/{new_name}")
        os.rename(f"cartosat/B/patch_{patch_num}_T2.tif", f"cartosat/B/{new_name}")
        os.rename(f"cartosat/mask/mask_{patch_num}.tif", f"cartosat/mask/{new_name}")

if __name__ == "__main__":
    rename_files()