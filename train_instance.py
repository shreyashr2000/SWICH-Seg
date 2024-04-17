import torch
from resnet_model import ResNet101
from resnet_gru_model import GRUClassifier
from resnet_gru_model import remove_last_fc_layer
from resnet_fc_model import ResNetFCModel
from unet_model import UNetModel
from data_loader_INSTANCE import get_INSTANCE_data_loader
from lowdata_class_trainer import train_model_lowdata
from gradcam_function import generate_gradcam
from clustering_function import perform_kmeans_clustering
from utils import your_loss_function, your_optimizer_function
from data_utils import class_lowdata_numpy_dataset
from data_utils import test_lowdata_numpy_dataset
from torch.utils.data import DataLoader
from loss_function import dice_loss
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
 #   optimizer = your_optimizer_function(resnet_model.parameters(), lr=learning_rate)  #  optimizer

    optimizer = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9)
    best_resnet_model = train_model_lowdata(resnet_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_model.pth')
    hidden_dim = 256  # Number of hidden units in the LSTM layer
    output_dim = 1  # Number of output classes (binary classification)

# Create an instance of the LSTM-based classifier
    # 2. Train ResNet-GRU model
    resnet_model=remove_last_fc_layer(resnet_model)   #Remove fc layr from resnet
    resnet_gru_model = GRUClassifier(resnet_model,hidden_dim, output_dim)  # Pass ResNet model as argument to ResNet-GRU model
    criterion = nn.BCELoss()   #  loss function
    learning_rate=0.001
    optimizer = optim.SGD(resnet_gru_model.parameters(), lr=learning_rate, momentum=0.9) #   optimizer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_resnet_gru_model = train_model_lowdata(resnet_gru_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_gru_model.pth')
    # Extract ResNet from ResNet-GRU model
    resnet_model_from_gru = best_resnet_gru_model.resnet()
    # 3. Train ResNet-FC model
    resnet_fc_model = ResNetFCModel(resnet_model_from_gru)  # Pass ResNet model as argument to ResNet-FC model
    criterion = nn.BCELoss()  #  loss function
    optimizer = optim.SGD(resnet_gru_model.parameters(), lr=learning_rate, momentum=0.9)  #   optimizer
    best_resnet_fc_model = train_model_lowdata(resnet_fc_model, train_loader, criterion, optimizer, device, epochs, 'best_resnet_fc_model.pth')
    # 4. Generate Grad-CAM
    # generates Grad-CAM images
    gradcam_images = generate_gradcam(best_resnet_fc_model, train_loader)

    # 5. Perform KMeans clustering
    cluster_labels = perform_kmeans_clustering(gradcam_images)

    # 6. Train UNet model using cluster labels
    unet_model = UNetModel()  # Initialize UNet model
    criterion = dice_loss()  #   loss function
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)  #   optimizer
    best_unet_model = train_model_lowdata(unet_model, train_loader, criterion, optimizer, device, epochs, 'best_unet_model.pth', cluster_labels=cluster_labels)

if __name__ == "__main__":
    main()
