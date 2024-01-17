import torch
import torchvision.models as models
import timm

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
    


class MyTimmNet:
    def __init__(self, train_loader, test_loader, num_classes, model_name='resnet18'):
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {100 * accuracy:.2f}%')


