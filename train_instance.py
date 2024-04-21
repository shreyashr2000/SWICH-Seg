import torch
from resnet_model import ResNet101
from resnet_gru_model import GRUClassifier
from resnet_gru_model import remove_last_fc_layer
from resnet_fc_model import ResNetFCModel
from UNet2D import UNet
from data_loader_INSTANCE import get_INSTANCE_data_loader
from lowdata_class_trainer import train_model_lowdata
from lowdata_seg_trainer import seg_train_model_lowdata
from gradcam_function import generate_gradcam
from clustering import perform_kmeans_clustering
from data_utils import class_lowdata_numpy_dataset
from data_utils import test_lowdata_numpy_dataset
from data_utils import seg_lowdata_numpy_dataset
from torch.utils.data import DataLoader
from loss_function import dice_loss
from generate_pseudolabels import get_pseudolabels
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import seaborn as sns
import glob
import cv2
import torch.optim as optim
def main():
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    epochs = 100
    # Load INSTANCE dataset
    train_loader = get_INSTANCE_data_loader(batch_size)
    # 1. Train ResNet model
    train_dataset = class_lowdata_numpy_dataset(train_data, train_labels)
    val_dataset = test_lowdata_numpy_dataset(val_data, val_labels)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    resnet_model = ResNet101()  # Initialize ResNet model
    criterion = nn.BCELoss()  #  loss function
    optimizer = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9) #optimizer
    best_resnet_model = train_model_lowdata(resnet_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_model.pth')
    hidden_dim = 256  # Number of hidden units in the LSTM layer
    output_dim = 1  # Number of output classes (binary classification)

# Create an instance of the LSTM-based classifier
    # 2. Train ResNet-GRU model
    best_resnet_model=remove_last_fc_layer(best_resnet_model)   #Remove fc layer from best_resnet_model
    resnet_gru_model = GRUClassifier(best_resnet_model,hidden_dim, output_dim)  # Pass Best_ResNet model as argument to ResNet-GRU model
    learning_rate=0.001
    optimizer = optim.SGD(resnet_gru_model.parameters(), lr=learning_rate, momentum=0.9) #   optimizer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Dont shuffle the train_loader, to give whole 3D scan as input to resnet-gru model.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_resnet_gru_model = train_model_lowdata(resnet_gru_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_gru_model.pth')
    # Extract ResNet from ResNet-GRU model
    resnet_model_from_gru = best_resnet_gru_model.resnet()  
    # 3. Train ResNet-FC model
    resnet_fc_model = ResNetFCModel(resnet_model_from_gru)  # Pass ResNet model as argument to ResNet-FC model
    optimizer = optim.SGD(resnet_gru_model.parameters(), lr=learning_rate, momentum=0.9)  #   optimizer
    best_resnet_fc_model = train_model_lowdata(resnet_fc_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_fc_model.pth')
    # 4. Generate pseudolabels-using GradCAM and KMeans clustering
    model=best_resnet_fc_model 
    model.eval()
    target_layer = model.module[-4] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_pseudolabels,val_pseudolabels = get_pseudolabels(train_loader,val_loader,model,device,target_layer)
    train_dataset = seg_lowdata_numpy_dataset(train_data, train_pseudolabels)
    val_dataset = test_lowdata_numpy_dataset(val_data, val_pseudolabels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 5. Train UNet model using cluster labels
    model = UNet(in_channels=1, out_channels=2, init_features=16)
    if cuda:
        model.cuda()
    criterion = dice_loss()  #   loss function
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)  #   optimizer
    best_unet_model = seg_train_model_lowdata(model, train_loader, criterion, optimizer, device, epochs, 'best_unet_model.pth')

if __name__ == "__main__":
    main()
