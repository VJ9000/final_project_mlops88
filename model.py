from torch import nn

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