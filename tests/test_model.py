import pytest
import torch
from training import VGGModel, VGGBlock  # Import VGGBlock directly

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model_construction():
    num_classes = 10
    model = VGGModel(classes_number=num_classes)
    model.to(device)
    
    # Check if the final fully connected layer outputs the correct number of classes
    assert model.fc3.out_features == num_classes, "The output features of the last layer should match the number of classes."
    
    # Check if the model is a subclass of nn.Module
    assert issubclass(type(model), torch.nn.Module), "The model should be a subclass of nn.Module."
    
    # Check the presence of certain layers (e.g., the first VGG block)
    assert hasattr(model, 'vgg_blocks'), "The model should have an attribute 'vgg_blocks'."
    assert isinstance(model.vgg_blocks[0], VGGBlock), "The first element of 'vgg_blocks' should be an instance of VGGBlock."
    
    # Check if the model is on the correct device
    assert next(model.parameters()).device == torch.device(device), f"The model parameters should be on the {device} device."