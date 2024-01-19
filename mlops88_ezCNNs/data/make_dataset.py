import os
import subprocess
import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


def check_dvc():
    try:
        subprocess.run(
            ["dvc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_dvc_pull():
    try:
        subprocess.run(["dvc", "pull"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running 'dvc pull': {e}")


def get_data_loaders(data_dir, batch_size=10, test_split=0.2):
    return create_data_loaders(data_dir, batch_size, test_split)


def create_data_loaders(data_root, batch_size=10, test_split=0.2):
    """
    Create training and test data loaders for a given dataset with class-wise splitting.

    Parameters:
    - data_root (str): Path to the root folder of the dataset.
    - batch_size (int): Batch size for the data loaders.
    - test_split (float): Fraction of data to be used for testing within each class.

    Returns:
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for test data.
    - class_names (list): The class names
    """

    # Define data transforms

    data_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    # Create dataset
    full_dataset = datasets.ImageFolder(root=data_root, transform=data_transform)

    # Create a dictionary to store indices for each class
    class_indices = {class_name: [] for class_name in full_dataset.classes}
    class_names = full_dataset.classes

    # Populate the dictionary with indices of images for each class
    for idx, (image_path, label) in enumerate(full_dataset.imgs):
        class_name = full_dataset.classes[label]
        class_indices[class_name].append(idx)

    # Split indices within each class
    train_indices, test_indices = [], []
    for class_name, indices in class_indices.items():
        split_idx = int(len(indices) * (1 - test_split))
        train_indices.extend(indices[:split_idx])
        test_indices.extend(indices[split_idx:])

    # Create data loaders based on the split indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )

    return train_loader, test_loader, class_names


def visualize_data(data_loader, class_names, num_images=5):
    """
    Visualize a few sample images along with their labels from a data loader.

    Parameters:
    - data_loader (DataLoader): DataLoader for the dataset.
    - class_names (list): List of class names.
    - num_images (int): Number of images to display for each batch.
    """

    for batch_images, batch_labels in data_loader:
        images = np.transpose(batch_images.numpy(), (0, 2, 3, 1))

        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].set_title(batch_labels[i])
            class_name = class_names[batch_labels[i]]
            axes[i].set_title(f"{class_name}")
            axes[i].axis("off")
        plt.show()

        break  # Display only the first batch for brevity


def visualize_class_distribution(data_root):
    # Assuming each subdirectory corresponds to a class
    classes = sorted(os.listdir(data_root))
    # Count the number of images in each class
    class_counts = {
        class_name: len(os.listdir(os.path.join(data_root, class_name)))
        for class_name in classes
    }
    # Plot the distribution
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Class")
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in all_image_data")
    plt.show()


# Helper function to calculate class counts from a DataLoader
def get_class_counts(loader):
    class_counts = {class_name: 0 for class_name in class_names}
    for _, batch_labels in loader:
        for label in batch_labels:
            class_counts[class_names[label]] += 1
    return class_counts


def visualize_sets_distribution(train_loader, test_loader, class_names):
    """
    Visualize the class distribution in the train set and test set.

    Parameters:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    - class_names (list): List of class names in the dataset.
    """

    # Calculate class counts for train and test sets
    train_class_counts = get_class_counts(train_loader)
    test_class_counts = get_class_counts(test_loader)

    # Plot the distribution for the train set
    plt.figure(figsize=(12, 6))
    plt.bar(train_class_counts.keys(), train_class_counts.values(), label="Train Set")
    plt.xlabel("Class")
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Train Set")
    plt.legend()
    plt.show()

    # Plot the distribution for the test set
    plt.figure(figsize=(12, 6))
    plt.bar(test_class_counts.keys(), test_class_counts.values(), label="Test Set")
    plt.xlabel("Class")
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Test Set")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Get the data and process it
    if not check_dvc():
        logging.error(
            "DVC is not installed.\n\
                       pip install dvc\npip install 'dvc[gdrive]'"
        )

    logging.info("Pulling data from DVC")
    run_dvc_pull()
    # For now we will not do any augmentation on the data
    train_set, test_set, class_names = create_data_loaders(DATA_ROOT_DIR)
    # visualize_sets_distribution(train_set,test_set,class_names)
