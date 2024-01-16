import torch
from torch import nn
from torch import optim
from collections import defaultdict
import time
from tqdm import tqdm
from IPython.display import clear_output
import gc
import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'

class VGGBlock(nn.Module):
    def __init__(self, layers_number, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(module=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), name='conv1')
        self.layers.add_module(module=nn.BatchNorm2d(out_channels), name='bn1')
        self.layers.add_module(module=nn.ReLU(), name='relu1')
        for i in range(layers_number - 1):
            self.layers.add_module(module=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   name=f'conv{i + 2}')
            self.layers.add_module(module=nn.BatchNorm2d(out_channels), name=f'bn{i + 2}')
            self.layers.add_module(module=nn.ReLU(), name=f'relu{i + 2}')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.pool(self.layers(x))
    

class VGGModel(nn.Module):
    def __init__(self, classes_number):
        super().__init__()
        self.classes_number = classes_number
        layers_structure = ((2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512))
        
        self.vgg_blocks = nn.Sequential()
        for i, layer_structure in enumerate(layers_structure):
            self.vgg_blocks.add_module(module=VGGBlock(*layer_structure), name=f'vgg{i + 1}')
            
        self.fltn = nn.Flatten()
        
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout()
        
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout()
        
        self.fc3 = nn.Linear(4096, classes_number)
        
    def forward(self, x):
        x = self.fltn(self.vgg_blocks(x))
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.do1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.do2(x)
        
        return self.fc3(x)

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

def train(model, criterion,optimizer, train_batch_gen,val_batch_gen=None,num_epochs=1,scheduler=None,history=None,checkpoint_path='state.pt',):
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