import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# Define the GRU-based classifier
class GRUClassifier(nn.Module):
    def __init__(self, resnet, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.resnet = resnet
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.GRU(input_size=num_ftrs, hidden_size=hidden_dim, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Pass the input through the ResNet
        features = self.resnet(x)
        # Resize the features to have batch size as the first dimension
        features = features.view(batch_size, 1, -1)
        # Pass the features through the GRU layer
        GRU_out, _ = self.lstm(features)
        # Pass the GRU output through the final classification layers
        output = F.relu(self.fc1(GRU_out[:, -1, :]))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
def remove_last_fc_layer(resnet):
    """
    Remove the last fully connected layer of a ResNet model.
    
    Args:
        resnet: Pre-trained ResNet model
        
    Returns:
        resnet: Modified ResNet model with the last fully connected layer removed
        num_ftrs: Number of input features for the removed fully connected layer
    """
    # Get the feature size of the last layer
    num_ftrs = resnet.fc.in_features
    
    # Remove the last fully connected layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    
    return resnet, num_ftrs
