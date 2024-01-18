import torch
import torchvision.models as models
from torch import nn
import timm
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl 
from torchvision.utils import make_grid
import wandb
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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
                wandb.log({f'Class_{class_name}_Accuracy': class_accuracy * 100})


        avg_acc = sum(class_accuracies) / len(class_accuracies)
        print(f'Average Test Accuracy: {avg_acc * 100:.2f}%')
        wandb.log({'Average_Test_Accuracy': avg_acc * 100})
        
        images, labels = next(iter(self.test_loader))
        predictions = torch.argmax(self(images), dim=1)
        class_names = [self.class_names[i] for i in predictions]

        grid = make_grid(images, nrow=4, normalize=True)
        grid_with_text = self.add_text_to_image_grid(grid, class_names)

        wandb.log({"Test Images": [wandb.Image(grid_with_text, caption="Test Images")]})
    
    def add_text_to_image_grid(self, grid, predictions):
        # Convert the PyTorch tensor to a PIL image
        pil_image = transforms.ToPILImage()(grid)

        # Create a drawing object
        draw = ImageDraw.Draw(pil_image)
        
        # Choose a font (you may need to adjust the path)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  
        font = ImageFont.truetype(font_path, size=14)

        # Add predictions as white text to the images
        for i, prediction in enumerate(predictions):
            x = (i % 4) * 128 + 10
            y = (i // 4) * 128 + 10
            draw.text((x, y), prediction, fill="blue", font=font)

        return pil_image