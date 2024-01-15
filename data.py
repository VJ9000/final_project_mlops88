import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

import torch
import torchvision
import torchvision.transforms as T
from torch import nn

import warnings
import gc
import os
import shutil
import time
from tqdm import tqdm
from IPython.display import clear_output

from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(12)
torch.manual_seed(12)

DATA_DIR = '../input/fruit-and-vegetable-image-recognition'
WORKING_DIR = './'
TRAIN_DIR = os.path.join(WORKING_DIR, 'train')
VAL_DIR = os.path.join(WORKING_DIR, 'val')
READ_TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

IS_TRAIN = True
WITH_VALIDATION = False

# creating directories for train and validation
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

CATEGORIES = {i: name for i, name in enumerate(os.listdir(READ_TRAIN_DIR))}
VAL_PART = 0.2 if WITH_VALIDATION else 0.0

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

weights = [100.0 / images_nums[category] for category in CATEGORIES.values()]

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

batch_size = 64

train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
if WITH_VALIDATION:
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    val_batch_gen = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
else:
    val_batch_gen = None



classes_number = len(CATEGORIES)
model = VGGModel(classes_number).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
if WITH_VALIDATION:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.01, factor=0.31, patience=7)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,35], gamma=0.31)



def plot_learning_curves(history):
    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('Loss', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    
    if 'val' in history['loss'].keys():
        plt.plot(history['loss']['val'], label='val')
        
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Accuracy', fontsize=15)
    plt.plot(history['accuracy']['train'], label='train')
    
    if 'val' in history['accuracy'].keys():
        plt.plot(history['accuracy']['val'], label='val')
        
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.legend()
    plt.show()

def train(
    model, 
    criterion,
    optimizer, 
    train_batch_gen,
    val_batch_gen=None,
    num_epochs=50,
    scheduler=None,
    history=None,
    checkpoint_path='state.pt',):
    if history is None:
        history = defaultdict(lambda: defaultdict(list))

    for epoch in range(num_epochs):
        train_loss = 0
        train_accuracy = 0
        val_loss = 0
        val_accuracy = 0
        
        start_time = time.time()

        model.train(True) 

        for X_batch, y_batch in tqdm(train_batch_gen, leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            
            loss = criterion(logits, y_batch.long().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_accuracy += np.mean(y_batch.cpu().numpy() == y_pred)

        train_loss /= len(train_batch_gen)
        train_accuracy /= len(train_batch_gen) 
        history['loss']['train'].append(train_loss)
        history['accuracy']['train'].append(train_accuracy)
        
        if val_batch_gen is not None:
            model.train(False)

            with torch.no_grad():
                for X_batch, y_batch in tqdm(val_batch_gen, leave=False):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch.long().to(device))
                    val_loss += np.sum(loss.detach().cpu().numpy())
                    y_pred = logits.max(1)[1].detach().cpu().numpy()
                    val_accuracy += np.mean(y_batch.cpu().numpy() == y_pred)

            val_loss /= len(val_batch_gen)
            val_accuracy /= len(val_batch_gen) 
            history['loss']['val'].append(val_loss)
            history['accuracy']['val'].append(val_accuracy)
        
        clear_output()

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss: \t\t\t{:.6f}".format(train_loss))
        
        if val_batch_gen is not None:
            print("  validation loss: \t\t\t{:.6f}".format(val_loss))
        
        print("  training accuracy: \t\t\t{:.2f} %".format(train_accuracy * 100))
        
        if val_batch_gen is not None:
            print("  validation accuracy: \t\t\t{:.2f} %".format(val_accuracy * 100))
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': dict(history),
        }
        
        if scheduler is not None:
            if val_batch_gen is None:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            state_dict['scheduler_state_dict'] = scheduler.state_dict()

        plot_learning_curves(history)
        torch.save(state_dict, checkpoint_path)
        gc.collect()
        
    return model, history


checkpoint_path = os.path.join(WORKING_DIR, 'vgg-adam-epoch50.pt')
if IS_TRAIN:
    model, history = train(model, criterion, optimizer, train_batch_gen, val_batch_gen,
                           scheduler=scheduler, checkpoint_path=checkpoint_path)


