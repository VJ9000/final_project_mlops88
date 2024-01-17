import os
import torch 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT_DIR = 'data/raw/all_image_data'

def fetch_data_with_dvc():
    try:
        import dvc
    except ImportError:
        print("dvc is not installed")
    # dvc needs to be in the path
    os.system("dvc pull")
    
    
def create_data_loaders(data_root, batch_size=32, test_split=0.2):
    """
    Create training and test data loaders for a given dataset.

    Parameters:
    - data_root (str): Path to the root folder of the dataset.
    - batch_size (int): Batch size for the data loaders.
    - test_split (float): Fraction of data to be used for testing.

    Returns:
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for test data.
    """

    # Define data transforms
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create dataset
    full_dataset = datasets.ImageFolder(root=data_root, transform=data_transform)

    # Calculate the number of samples for training and testing
    num_samples = len(full_dataset)
    num_test_samples = int(test_split * num_samples)
    num_train_samples = num_samples - num_test_samples

    # Split the dataset into training and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [num_train_samples, num_test_samples])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_data(data_loader, class_names=None, num_images=5):
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
            axes[i].axis('off')
        plt.show()

        break  # Display only the first batch for brevity

def visualize_class_distribution(data_root):
    # Assuming each subdirectory corresponds to a class
    classes = sorted(os.listdir(data_root))
    # Count the number of images in each class
    class_counts = {class_name: len(os.listdir(os.path.join(data_root, class_name))) for class_name in classes}
    # Plot the distribution
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.xticks(rotation='vertical')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in all_image_data')
    plt.show()

if __name__ == '__main__':
    # Get the data and process it
    # For now we will not do any augmentation on the data 
    
    visualize_class_distribution(DATA_ROOT_DIR)
    train_set,test_set = create_data_loaders(DATA_ROOT_DIR)
    # print(train_set.dataset.classes)
    visualize_data(train_set,num_images = 10)
    
    
    