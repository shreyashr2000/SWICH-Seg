import torch
from resnet_gru_model import GRUClassifier
from resnet_gru_model import remove_last_fc_layer
from resnet_fc_model import ResNetFCModel
from UNet2D import UNet
from loader import data_load
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
from inference_unet_model import unet_test
from metrics import MetricsCalculator
from loader import data_load
def main():
    dice=0
    tpr=0
    iou=0
    count=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    epochs = 100
    # Load INSTANCE dataset
    test_data,test_mask = data_load('','',large=False,train=False)
    test_dataset = test_lowdata_numpy_dataset(test_data, test_labels)
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    resnet_gru_model = GRUClassifier(best_resnet_model,hidden_dim, output_dim)
    LOAD_PATH = 'best_resnet_gru_model.pth'  ###output neuron is 1  ##main_output
    
    if cuda:
      resnet_gru_model.cuda()
    model_dict = resnet_gru_model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
   # print('weights loaded model = ', len(pretrained_dict), '/', len(model_dict))
    resnet_gru_model.load_state_dict(torch.load(LOAD_PATH))
    resnet_model_from_gru = resnet_gru_model.resnet()  
    resnet_fc_model = ResNetFCModel(resnet_model_from_gru) 
    LOAD_PATH = 'best_resnet_fc_model.pth'  ###output neuron is 1  ##main_output
    if cuda:
      resnet_fc_model.cuda()
    model_dict = resnet_fc_model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model=best_resnet_fc_model 
    model.eval()
    target_layer = model.module[-4] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_loader = DataLoader(torch.empty(0), batch_size=1)
# Assuming model, device, and target_layer are defined elsewhere
    pseudolabels, _ = get_pseudolabels(test_loader, val_loader, model, device, target_layer)
    model = UNet(in_channels=1, out_channels=2, init_features=16)

    LOAD_PATH = 'best_unet_model.pth'  ###output neuron is 1
    if cuda:
      model.cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(LOAD_PATH)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #print('weights loaded model = ', len(pretrained_dict), '/', len(model_dict))
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()
    pseudo_unet_mask=unet_test(model,test_dataloader,device)
    final_output=np.zeros((pseudolabels.shape[0],512,512))
    metrics_calculator = MetricsCalculator()


    for i in range(len(pseudolabels)):
        final_output[i,:,:]=np.logical_or(pseudolabels[i,:,:],pseudo_unet_mask[i,:,:])
        if test_mask[i,1,:,:].max()>0:
            dice_temp, tpr_temp, iou_temp = metrics_calculator.calculate_all_metrics()
            dice += dice_temp
            tpr += tpr_temp
            iou += iou_temp
            count=count+1
    final_dice=dice/count
    final_tpr=tpr/count   
    final_iou=iou/count
    print("Dice Coefficient:", final_dice)
    print("True Positive Rate (TPR):", final_tpr)
    print("Intersection over Union (IoU):", final_iou)
        
  
    
