import torch.nn as nn
import torch.nn.functional as F
import torch
from Unet_parts import *
import json


with open("data\\uptown_funk.json","r") as f:
    data = json.load(f)

def load_data(data : dict,num_samples: int):
    for _ in range(len(data["nose"])//num_samples):
        joint_positions = [data[joint][:num_samples]for joint in data.keys()]
        tensor_data = torch.tensor(joint_positions)
        tensor_data= tensor_data.permute(dims=(1,0,2))
        
        yield tensor_data

# y = load_data(data,10)
# z  = next(y)

# print(z.size())   #image_size = (num_samples, 20, 2 )

class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.up4 = (Up(128, 64))
        self.outc = (DoubleConv(64, n_channels))

    def forward(self, x , t):
        x1 = self.inc(x,t)
        x2 = self.down1(x1,t)
        x = self.up4(x2, x1,t)
        out = self.outc(x,t)
        return out




# y = next(load_data(data,10))
# y = y.unsqueeze(0)
# print(y.size())
# z = UNet(10)
# print(z(y,2))
