import os
import random

def create_list_files():
    # Define the cartosat dataset path
    cartosat_path = "cartosat"
    A_path = os.path.join(cartosat_path, "A")
    list_path = os.path.join(cartosat_path, "list")
    
    # Check if the A directory exists
    if not os.path.exists(A_path):
        print(f"Error: Directory {A_path} not found!")
        return
    
    # Create list directory if it doesn't exist
    os.makedirs(list_path, exist_ok=True)
    
    # Get all image files from the A directory
    image_files = [f for f in os.listdir(A_path) if f.endswith('.png')]
    
    if not image_files:
        print(f"Error: No .tif files found in {A_path}")
        return
    
    # Shuffle for random split
    random.shuffle(image_files)
    
    total_images = len(image_files)
    
    # Define split percentages (adjust as needed)
    train_sup_ratio = 0.6    # 60% for supervised training
    train_unsup_ratio = 0.2  # 20% for unsupervised training  
    val_ratio = 0.1          # 10% for validation
    test_ratio = 0.1         # 10% for testing
    
    # Calculate split indices
    train_sup_end = int(total_images * train_sup_ratio)
    train_unsup_end = train_sup_end + int(total_images * train_unsup_ratio)
    val_end = train_unsup_end + int(total_images * val_ratio)
    
    # Split the files
    train_sup_files = image_files[:train_sup_end]
    train_unsup_files = image_files[train_sup_end:train_unsup_end]
    val_files = image_files[train_unsup_end:val_end]
    test_files = image_files[val_end:]
    
    # Write list files
    def write_list_file(filename, file_list):
        filepath = os.path.join(list_path, filename)
        with open(filepath, 'w') as f:
            for file in file_list:
                f.write(f"{file}\n")
        print(f"Created: {filepath}")
    
    write_list_file("train_supervised.txt", train_sup_files)
    write_list_file("train_unsupervised.txt", train_unsup_files)
    write_list_file("val.txt", val_files)
    write_list_file("test.txt", test_files)
    
    print(f"\nDataset split summary:")
    print(f"Total images: {total_images}")
    print(f"Train supervised: {len(train_sup_files)}")
    print(f"Train unsupervised: {len(train_unsup_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    
    # Show some example files from each split
    print(f"\nExample files from each split:")
    if train_sup_files:
        print(f"Train supervised example: {train_sup_files[0]}")
    if train_unsup_files:
        print(f"Train unsupervised example: {train_unsup_files[0]}")
    if val_files:
        print(f"Validation example: {val_files[0]}")
    if test_files:
        print(f"Test example: {test_files[0]}")

if __name__ == "__main__":
    create_list_files()