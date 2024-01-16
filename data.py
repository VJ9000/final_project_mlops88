import os
import shutil
import numpy as np
import torchvision.transforms as T

# Data preparation and loading functions
DATA_DIR = './files'
WORKING_DIR = './'
TRAIN_DIR = os.path.join(WORKING_DIR, 'train')
VAL_DIR = os.path.join(WORKING_DIR, 'val')
READ_TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

WITH_VALIDATION = False

CATEGORIES = {i: name for i, name in enumerate(os.listdir(READ_TRAIN_DIR))}
VAL_PART = 0.2 if WITH_VALIDATION else 0.0

def prepare_data_dirs():
    # creating directories for train and validation
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # dividing the training sample into training and validation samples
    images_nums = {}
    for category in CATEGORIES.values():
        # preparing training and validation directories in working directory
        train_category_dir = os.path.join(TRAIN_DIR, category)
        val_category_dir = os.path.join(VAL_DIR, category)
        
        shutil.rmtree(val_category_dir, ignore_errors=True)
        shutil.rmtree(train_category_dir, ignore_errors=True)
        os.makedirs(val_category_dir, exist_ok=True)
        shutil.copytree(os.path.join(READ_TRAIN_DIR, category), train_category_dir)
        
        # generating numbers of images for validation
        images_names = sorted(os.listdir(train_category_dir))
        images_num = len(images_names)
        val_images_num = int(images_num * VAL_PART)
        val_images_names = np.take(images_names, np.random.choice(images_num, val_images_num, replace=False))
        images_nums[category] = images_num
        
        # copy needed images to validation directory and remove them from training directiry
        for image_name in val_images_names:
            cur_image = os.path.join(train_category_dir, image_name)
            shutil.copy(cur_image, os.path.join(os.path.join(VAL_DIR, category), image_name))
            os.remove(cur_image)
        
        print(f'{category}: train_images = {images_num - val_images_num}, val_images = {val_images_num}')

    return images_nums

IMG_SIZE = (224, 224)

# augumentations
train_transforms = T.Compose([
    T.Resize(IMG_SIZE),
    T.RandomAffine(20),
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.2)),
    T.ToTensor(),
    T.Lambda(lambda x: x[:3])
])

val_transforms = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Lambda(lambda x: x[:3])
])