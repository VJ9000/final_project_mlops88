import torch 
from torch import nn
import torchvision
import numpy as np
import os
import data
import training
import testing


device = 'cuda' if torch.cuda.is_available() else 'cpu'
WITH_VALIDATION = False

np.random.seed(12)
torch.manual_seed(12)

IS_TRAIN = True
images_nums = data.prepare_data_dirs()
weights = [100.0 / images_nums[category] for category in data.CATEGORIES.values()]
batch_size = 64
train_dataset = torchvision.datasets.ImageFolder(data.TRAIN_DIR, transform=data.train_transforms)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_batch_gen = None
if WITH_VALIDATION:
   val_dataset = torchvision.datasets.ImageFolder(data.VAL_DIR, transform=data.val_transforms)
   val_batch_gen = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
classes_number = len(data.CATEGORIES)
model = training.VGGModel(classes_number).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = None
if WITH_VALIDATION:
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.01, factor=0.31, patience=7)
else:
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,35], gamma=0.31)
checkpoint_path = os.path.join(data.WORKING_DIR, 'vgg-adam-epoch50.pt')
if IS_TRAIN:
   model, history = training.train(model, criterion, optimizer, train_batch_gen, val_batch_gen, scheduler=scheduler, checkpoint_path=checkpoint_path)
test_dataset = torchvision.datasets.ImageFolder(data.TEST_DIR, transform=data.val_transforms)
test_batch_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
state = torch.load(checkpoint_path)
model.load_state_dict(state['model_state_dict'])
testing.test_model(model, test_batch_gen)