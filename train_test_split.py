import os
import shutil
import random

src_folder = "./data/Images"
train_folder = "./data/train"
test_folder = "./data/test"

# Set random seed for reproducibility
random.seed(42)

# Set the ratio of training set
train_ratio = 0.8

# Iterate through subfolders in the source folder
for subdir, _, files in os.walk(src_folder):
    if subdir == src_folder:
        continue

    # Get the subfolder name
    folder_name = os.path.basename(subdir)

    # Create corresponding subfolders in train and test folders
    os.makedirs(os.path.join(train_folder, folder_name), exist_ok=True)
    os.makedirs(os.path.join(test_folder, folder_name), exist_ok=True)

    # Randomly shuffle the file list
    random.shuffle(files)

    # Calculate the size of the training set
    train_size = int(train_ratio * len(files))

    # Assign files to training and test sets
    train_files = files[:train_size]
    test_files = files[train_size:]

    # Copy images to corresponding train and test folders
    for file in train_files:
        shutil.copy(os.path.join(subdir, file), os.path.join(train_folder, folder_name, file))

    for file in test_files:
        shutil.copy(os.path.join(subdir, file), os.path.join(test_folder, folder_name, file))

print("Image splitting complete!")
