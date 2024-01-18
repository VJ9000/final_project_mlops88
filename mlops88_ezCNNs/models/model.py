import torch
import torchvision.models as models
from torch import nn
import timm
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl 

class MyTimmNet(pl.LightningModule):
    def __init__(self, train_loader, test_loader, class_names, model_name='resnet18', learning_rate=0.001):
        super(MyTimmNet, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True)
        self.num_classes = len(class_names)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.accuracy = Accuracy(num_classes=self.num_classes,task='multiclass')
        self.class_names = class_names
        self.test_results = {'outputs': [], 'labels': []}
        self.learning_rate = learning_rate


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
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
        self.test_results['outputs'].append(outputs)
        self.test_results['labels'].append(labels)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        return acc


    def on_test_epoch_end(self):
        all_outputs = torch.cat(self.test_results['outputs'])
        all_labels = torch.cat(self.test_results['labels'])
        class_accuracies = []
        for class_name in self.class_names:
            class_mask = (all_labels == self.class_names.index(class_name))
            class_outputs = all_outputs[class_mask]
            class_labels = all_labels[class_mask]
            
            if len(class_labels) > 0:
                class_accuracy = torch.sum(torch.argmax(class_outputs, dim=1) == class_labels).item() / len(class_labels)
                class_accuracies.append(class_accuracy)

                print(f'Class {class_name} Accuracy: {class_accuracy * 100:.2f}%')

        avg_acc = sum(class_accuracies) / len(class_accuracies)
        print(f'Average Test Accuracy: {avg_acc * 100:.2f}%')
