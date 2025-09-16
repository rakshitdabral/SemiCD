import os

# Base folder
base_dir = "mask"

# Walk through all subdirectories
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root, file)
            print(f"Deleting: {file_path}")
            os.remove(file_path)

print("All .xml files removed successfully.")
