import torch.nn as nn
from torchvision import models

class ResNet101:
    def __init__(self):
        # Initialize the ResNet model
        self.resnet = models.resnet101(pretrained=True)
        
        # Get the number of input features for the fully connected layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the fully connected layer with a new one with output size 1
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        return self.resnet(x)
