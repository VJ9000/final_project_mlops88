import torch
import torchvision.models as models
from torch import nn
import timm
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        self.l1 = torch.nn.Linear(in_features, 500)
        self.l2 = torch.nn.Linear(500, out_features)
        self.r = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))
    


class MyTimmNet(pl.LightningModule):
    def __init__(self, train_loader, test_loader, num_classes, model_name='resnet18'):
        super(MyTimmNet, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.accuracy = Accuracy(num_classes=num_classes,task='multiclass')
        self.test_results = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return acc

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Append the test result to the list
        self.test_results.append(acc)

        return acc

    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_results).mean()
        print(f'Test Accuracy: {avg_acc.item() * 100:.2f}%')

