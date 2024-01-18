import pytest
import torch
from mlops88_ezCNNs.models.model import MyTimmNet

@pytest.fixture
def my_timm_net():
    # Create an instance of MyTimmNet with default parameters
    net = MyTimmNet()
    return net

def test_forward_pass(my_timm_net):
    # Create a dummy batch of images (batch size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Perform a forward pass using the dummy input
    output = my_timm_net(dummy_input)
    
    # Check if the output shape is as expected (batch size, number of classes)
    assert output.shape == (1, my_timm_net.num_classes), "Output shape is incorrect"