import os

# Path to your folder
folder_path = "mask"

for filename in os.listdir(folder_path):
    if filename.startswith("mask_") and filename.endswith(".tif"):
        # Remove the "mask_" prefix
        new_name = filename.replace("mask_", "", 1)
        
        # Get full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
