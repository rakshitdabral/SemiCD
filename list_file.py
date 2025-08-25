import os
import random

def create_list_files():
    # Define the cartosat dataset path
    cartosat_path = "cartosat"
    A_path = os.path.join(cartosat_path, "A")
    label_path = os.path.join(cartosat_path, "label")
    list_path = os.path.join(cartosat_path, "list")
    
    # Check if the directories exist
    if not os.path.exists(A_path):
        print(f"Error: Directory {A_path} not found!")
        return
    
    if not os.path.exists(label_path):
        print(f"Error: Directory {label_path} not found!")
        return
    
    # Create list directory if it doesn't exist
    os.makedirs(list_path, exist_ok=True)
    
    # Get all image files from the A directory
    all_image_files = [f for f in os.listdir(A_path) if f.endswith('.png')]
    
    # Get all label files
    label_files = [f for f in os.listdir(label_path) if f.endswith('.png')]
    
    if not all_image_files:
        print(f"Error: No .png files found in {A_path}")
        return
    
    if not label_files:
        print(f"Error: No .png files found in {label_path}")
        return
    
    # Separate images with labels from those without labels
    images_with_labels = []
    images_without_labels = []
    
    for img_file in all_image_files:
        if img_file in label_files:
            images_with_labels.append(img_file)
        else:
            images_without_labels.append(img_file)
    
    print(f"Total images: {len(all_image_files)}")
    print(f"Images with labels: {len(images_with_labels)}")
    print(f"Images without labels: {len(images_without_labels)}")
    
    # For quick testing, limit the dataset size
    MAX_LABELED = 20  # Use only 20 labeled images
    MAX_UNLABELED = 50  # Use only 50 unlabeled images
    
    # Shuffle and limit labeled images
    random.shuffle(images_with_labels)
    images_with_labels = images_with_labels[:MAX_LABELED]
    
    # Shuffle and limit unlabeled images
    random.shuffle(images_without_labels)
    images_without_labels = images_without_labels[:MAX_UNLABELED]
    
    # Split labeled images for supervised training
    total_labeled = len(images_with_labels)
    
    # Define split percentages for labeled images
    train_sup_ratio = 0.7    # 70% for supervised training
    val_ratio = 0.15         # 15% for validation
    test_ratio = 0.15        # 15% for testing
    
    # Calculate split indices
    train_sup_end = int(total_labeled * train_sup_ratio)
    val_end = train_sup_end + int(total_labeled * val_ratio)
    
    # Split the labeled files
    train_sup_files = images_with_labels[:train_sup_end]
    val_files = images_with_labels[train_sup_end:val_end]
    test_files = images_with_labels[val_end:]
    
    # Use limited unlabeled images for unsupervised training
    train_unsup_files = images_without_labels
    
    # Write list files
    def write_list_file(filename, file_list):
        filepath = os.path.join(list_path, filename)
        with open(filepath, 'w') as f:
            for file in file_list:
                f.write(f"{file}\n")
        print(f"Created: {filepath} with {len(file_list)} files")
    
    write_list_file("train_supervised.txt", train_sup_files)
    write_list_file("train_unsupervised.txt", train_unsup_files)
    write_list_file("val.txt", val_files)
    write_list_file("test.txt", test_files)
    
    # Also create the percentage-based files
    write_list_file("100_train_supervised.txt", train_sup_files)
    write_list_file("100_train_unsupervised.txt", train_unsup_files)
    
    print(f"\nDataset split summary (QUICK TEST VERSION):")
    print(f"Total images: {len(all_image_files)}")
    print(f"Images with labels: {len(images_with_labels)} (limited to {MAX_LABELED})")
    print(f"Images without labels: {len(images_without_labels)} (limited to {MAX_UNLABELED})")
    print(f"Train supervised: {len(train_sup_files)} (from labeled images)")
    print(f"Train unsupervised: {len(train_unsup_files)} (from unlabeled images)")
    print(f"Validation: {len(val_files)} (from labeled images)")
    print(f"Test: {len(test_files)} (from labeled images)")
    
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