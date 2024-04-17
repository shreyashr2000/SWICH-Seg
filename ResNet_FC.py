import torch.nn as nn

class CustomResNetWithFC(nn.Module):
    def __init__(self, lstm_classifier):
        super(CustomResNetWithFC, self).__init__()
        self.resnet = lstm_classifier.resnet
        
        # Define the sequential container for ResNet with custom fully connected layer
        self.resnet_with_fc = nn.Sequential()
        
        # Add layers of ResNet (including all layers except the last one)
        for layer in list(self.resnet.children()):
            self.resnet_with_fc.add_module(str(len(self.resnet_with_fc)), layer)
        
        # Add a reshape layer to reshape features before the fully connected layer
        self.resnet_with_fc.add_module("reshape", Reshape())
        
        # Add a custom fully connected layer
        self.resnet_with_fc.add_module("fc", nn.Linear(in_features=2048, out_features=1))
        
        # Freeze all parameters except the parameters of the custom fully connected layer
        for param in self.resnet_with_fc.parameters():
            param.requires_grad = False
        
        # Set requires_grad to True for parameters of the custom fully connected layer
        for param in self.resnet_with_fc.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet_with_fc(x)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
